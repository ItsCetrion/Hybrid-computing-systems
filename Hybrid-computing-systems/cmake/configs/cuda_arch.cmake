function(set_target_minimum_cuda_arch target arch)
    if(NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH ${arch})
    endif()

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
        >
    )
endfunction()