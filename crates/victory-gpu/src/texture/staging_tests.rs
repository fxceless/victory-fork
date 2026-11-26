use std::cell::RefCell;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use super::{STAGING, Staging, StagingEntry, host, is_staged, staging, unstage};
use crate::object::{This, guid, slot};
use crate::resource::{HostTexture, ResourceDesc, create_at_address, host_copy, texture_layout};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Point {
    Creating,
    Waiting,
    RemovalWaiting,
    RemovalReady,
}

type Hook = (Point, Box<dyn FnOnce() -> bool>);
thread_local! {
    static HOOKS: RefCell<Vec<Hook>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn at(point: Point) -> bool {
    let hook = HOOKS.with(|hooks| {
        let mut hooks = hooks.borrow_mut();
        let index = hooks.iter().position(|(p, _)| *p == point)?;
        Some(hooks.remove(index).1)
    });
    hook.is_some_and(|call| call())
}

pub(super) fn create(make: impl FnOnce() -> Option<Arc<Staging>>) -> Option<Arc<Staging>> {
    if at(Point::Creating) { None } else { make() }
}

const TIMEOUT: Duration = Duration::from_secs(5);

fn receive<T>(rx: &Receiver<T>) -> T {
    rx.recv_timeout(TIMEOUT).expect("staging test checkpoint")
}

// Keep the fixture owned by each worker. A failing timeout must not join
// a thread stuck inside the code under test.
fn worker<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Receiver<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let value = work();
        let _ = tx.send(value);
    });
    rx
}

struct Checkpoint {
    reached: Receiver<()>,
    resume: Sender<()>,
}

fn pause(
    point: Point,
    action: impl FnOnce() -> bool + Send + 'static,
) -> (Checkpoint, impl FnOnce()) {
    let (reached_tx, reached) = mpsc::channel();
    let (resume, resume_rx) = mpsc::channel();
    let install = move || {
        HOOKS.with(|hooks| {
            hooks.borrow_mut().push((
                point,
                Box::new(move || {
                    reached_tx.send(()).unwrap();
                    receive(&resume_rx);
                    action()
                }),
            ));
        });
    };
    (Checkpoint { reached, resume }, install)
}

fn observe(point: Point) -> (Receiver<()>, impl FnOnce()) {
    let (tx, rx) = mpsc::channel();
    let install = move || {
        HOOKS.with(|hooks| {
            hooks.borrow_mut().push((
                point,
                Box::new(move || {
                    tx.send(()).unwrap();
                    false
                }),
            ));
        });
    };
    (rx, install)
}

struct Texture {
    resource: usize,
    host: Arc<HostTexture>,
    _memory: Box<[u8]>,
}

impl Texture {
    fn new() -> Arc<Self> {
        // Synthetic RGBA8 16x4, one mip and slice. The descriptor is the
        // existing 56-byte C layout, including padding after dimension.
        let words = [3u32, 0, 0, 0, 16, 0, 4, 0x0001_0001, 28, 1, 0, 256, 0, 0];
        // SAFETY: all fields in ResourceDesc are integers with the C layout above.
        let desc: ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = texture_layout(&desc).expect("synthetic layout");
        let memory = vec![0; layout.planes[0].layout.size as usize].into_boxed_slice();
        let iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut resource = std::ptr::null();
        assert_eq!(
            create_at_address(
                std::ptr::null(),
                memory.as_ptr() as u64,
                &desc,
                0,
                std::ptr::null(),
                &iid,
                &mut resource
            ),
            0,
        );
        let host = host_copy(resource).expect("synthetic host texture");
        Arc::new(Self {
            resource: resource as usize,
            host,
            _memory: memory,
        })
    }

    fn stage(&self) -> Option<Arc<Staging>> {
        staging(self.resource as This, &self.host.texture)
    }

    fn staged(&self) -> bool {
        is_staged(self.resource as This)
    }

    fn remove(&self) {
        unstage(self.resource as This);
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        let resource = self.resource as This;
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource),
            0
        );
    }
}

fn check_table() {
    let table = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let table = table.as_ref().expect("staging table exists");
    let pending = table
        .entries
        .values()
        .filter(|entry| matches!(entry, StagingEntry::Creating(_)))
        .count();
    assert_eq!(table.creating, pending);
}

#[test]
fn independent_progress_and_one_creator_per_identity() {
    if host().is_none() {
        eprintln!("no D3D12 host for staging test");
        return;
    }
    let held = Texture::new();
    let hot = Texture::new();
    let hot_staging = hot.stage().unwrap();
    let cold = Texture::new();
    let (creation, install) = pause(Point::Creating, || false);
    let creator = worker({
        let held = held.clone();
        move || {
            install();
            held.stage().unwrap()
        }
    });
    receive(&creation.reached);

    let (waiting, install) = observe(Point::Waiting);
    let same = worker(move || {
        install();
        held.stage().unwrap()
    });
    receive(&waiting);
    let independent = worker(move || {
        check_table();
        assert!(hot.staged());
        assert!(Arc::ptr_eq(&hot.stage().unwrap(), &hot_staging));
        assert!(cold.stage().is_some());
    });
    receive(&independent);
    creation.resume.send(()).unwrap();
    assert!(Arc::ptr_eq(&receive(&creator), &receive(&same)));
    check_table();
}

#[test]
fn failed_and_panicking_creators_wake_waiters_and_allow_retry() {
    if host().is_none() {
        eprintln!("no D3D12 host for staging test");
        return;
    }
    for unwind in [false, true] {
        let texture = Texture::new();
        let (creation, install) = pause(Point::Creating, move || {
            assert!(!unwind, "injected creator panic");
            true
        });
        let creator = worker({
            let texture = texture.clone();
            move || {
                install();
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| texture.stage().is_some()))
                    .map_err(|_| ())
            }
        });
        receive(&creation.reached);
        let (waiting, install) = observe(Point::Waiting);
        let stage_waiter = worker({
            let texture = texture.clone();
            move || {
                install();
                texture.stage().is_some()
            }
        });
        receive(&waiting);
        let (waiting, install) = observe(Point::Waiting);
        let read_waiter = worker({
            let texture = texture.clone();
            move || {
                install();
                texture.staged()
            }
        });
        receive(&waiting);

        creation.resume.send(()).unwrap();
        assert_eq!(receive(&creator), if unwind { Err(()) } else { Ok(false) });
        assert!(!receive(&stage_waiter));
        assert!(!receive(&read_waiter));
        check_table();
        let retry = worker(move || {
            assert!(!texture.staged());
            assert!(texture.stage().is_some());
        });
        receive(&retry);
        check_table();
    }
}

#[test]
fn concurrent_removal_only_erases_its_own_generation() {
    if host().is_none() {
        eprintln!("no D3D12 host for staging test");
        return;
    }
    let texture = Texture::new();
    let (creation, install) = pause(Point::Creating, || false);
    let creator = worker({
        let texture = texture.clone();
        move || {
            install();
            texture.stage().unwrap()
        }
    });
    receive(&creation.reached);

    let (waiting, install_wait) = observe(Point::RemovalWaiting);
    let (comparison, install_comparison) = pause(Point::RemovalReady, || false);
    let delayed = worker({
        let texture = texture.clone();
        move || {
            install_wait();
            install_comparison();
            texture.remove();
        }
    });
    receive(&waiting);
    let (waiting, install) = observe(Point::RemovalWaiting);
    let immediate = worker({
        let texture = texture.clone();
        move || {
            install();
            texture.remove();
            assert!(!texture.staged());
        }
    });
    receive(&waiting);

    creation.resume.send(()).unwrap();
    let old = receive(&creator);
    receive(&comparison.reached);
    receive(&immediate);
    let replacement = worker({
        let texture = texture.clone();
        move || texture.stage().unwrap()
    });
    let replacement = receive(&replacement);
    assert!(!Arc::ptr_eq(&old, &replacement));
    comparison.resume.send(()).unwrap();
    receive(&delayed);
    let visible = worker(move || texture.stage().unwrap());
    assert!(Arc::ptr_eq(&replacement, &receive(&visible)));
    check_table();
}
