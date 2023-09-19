pub fn add_to_all<const SIZE: usize>(
    a: &mut [f32; SIZE],
    b: &[f32],
) {
    #[cfg(feature = "simd")]
    {
        #[cfg(target_feature = "avx512f")]
        unsafe { add_to_all_avx512(a, b); }
        #[cfg(not(target_feature = "avx512f"))]
        add_to_all_base(a, b);
    }
    #[cfg(not(feature = "simd"))]
    add_to_all_base(a, b);
}

fn add_to_all_base<const SIZE: usize>(
    a: &mut [f32; SIZE],
    b: &[f32]
) {
    for (i, &j) in a.iter_mut().zip(b.iter()) {
        *i += j;
    }
}

#[cfg(feature = "simd")]
#[cfg(target_feature = "avx512f")]
unsafe fn add_to_all_avx512<const SIZE: usize>(
    a: &mut [f32; SIZE],
    b: &[f32],
) {
    use std::arch::x86_64::{
        _mm512_loadu_ps,
        _mm512_add_ps,
        _mm512_storeu_ps,
    };

    for (i, j) in a.chunks_exact_mut(64).zip(b.chunks_exact(64)) {
        let aptr0 = i.as_mut_ptr();
        let aptr1 = aptr0.wrapping_add(16);
        let aptr2 = aptr0.wrapping_add(32);
        let aptr3 = aptr0.wrapping_add(48);
        let bptr = j.as_ptr();

        let mut zmm0 = _mm512_loadu_ps(aptr0);
        let mut zmm1 = _mm512_loadu_ps(aptr1);
        let mut zmm2 = _mm512_loadu_ps(aptr2);
        let mut zmm3 = _mm512_loadu_ps(aptr3);

        zmm0 = _mm512_add_ps(zmm0, _mm512_loadu_ps(bptr));
        zmm1 = _mm512_add_ps(zmm1, _mm512_loadu_ps(bptr.wrapping_add(16)));
        zmm2 = _mm512_add_ps(zmm2, _mm512_loadu_ps(bptr.wrapping_add(32)));
        zmm3 = _mm512_add_ps(zmm3, _mm512_loadu_ps(bptr.wrapping_add(48)));

        _mm512_storeu_ps(aptr0, zmm0);
        _mm512_storeu_ps(aptr1, zmm1);
        _mm512_storeu_ps(aptr2, zmm2);
        _mm512_storeu_ps(aptr3, zmm3);
    }
}