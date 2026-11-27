    /// The patch's source compiles as the compare's does, with the same root
    /// signature inside it (no device needed).
    #[test]
    fn the_masked_patch_compiles() {
        let code = super::masked_patch_code().expect("MASKED_PATCH compiles");
        // SAFETY: the blob's own bytes.
        let size = unsafe { code.GetBufferSize() };
        assert!(size > 100);
    }
