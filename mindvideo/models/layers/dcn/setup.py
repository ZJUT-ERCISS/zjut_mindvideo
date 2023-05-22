# encoding: utf-8
import sys
import os
from setuptools import setup
from torch.utils import cpp_extension

def main():
    cpp_name = sys.argv[1]
    file_name, suffix = cpp_name.split('.')
    so_name = sys.argv[2]
    sys.argv[1:] = ['build_ext', '-i']
    if suffix in ['cpp', 'cc', 'c']:
        setup(
                name=file_name,                            # 编译后的链接库名称
                ext_modules=[
                    cpp_extension.CppExtension(
                        name=file_name,
                        sources=[cpp_name, 'ms_ext.cpp'],  # 待编译文件
                        extra_compile_args=[]
                        )
                    ],
                cmdclass={                                 # 执行编译命令设置
                    'build_ext': cpp_extension.BuildExtension
                    }
                )
    if suffix in ['cu']:
        setup(
                name=file_name,                            # 编译后的链接库名称
                ext_modules=[
                    cpp_extension.CUDAExtension(
                        name=file_name,
                        sources=[cpp_name, 
                                 'ms_ext.cpp',
                                 'deform_conv.cpp',
                                 'deform_conv_cuda.cu',
                                 'deform_conv_cuda_kernel.cu'],  # 待编译文件
                        define_macros = [("WITH_CUDA", None)],
                        extra_compile_args={"nvcc":[
                            "-O3",
                            "-DCUDA_HAS_FP16=1",
                            "-D__CUDA_NO_HALF_OPERATORS__",
                            "-D__CUDA_NO_HALF_CONVERSIONS__",
                            "-D__CUDA_NO_HALF2_OPERATORS__"],
                            "cxx":[],}
                        )
                    ],
                cmdclass={                                 # 执行编译命令设置
                    'build_ext': cpp_extension.BuildExtension
                    }
                )
    files = os.listdir(".")
    old_name = None
    for f in files:
        if f.startswith(file_name) and f.endswith('.so'):
            old_name = f
    if old_name:
        os.rename(old_name, so_name)

if __name__ == "__main__":
    main()
