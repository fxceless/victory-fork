// VICTORY_BRIGHTEN: the presented image through a gamma curve and a gain, so
// the dark parts (light the host does not draw yet) can be seen.
// Built with: dxc -T cs_6_0 -E main -Fo brighten.cso brighten.hlsl
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0), UAV(u0))"

cbuffer Curve : register(b0)
{
    float gamma; // below 1 lifts dark tones
    float gain;  // after the curve
};

Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    src.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    float4 c = src[id.xy];
    c.rgb = saturate(pow(max(c.rgb, 0.0), gamma) * gain);
    dst[id.xy] = c;
}
