#!/usr/bin/env python
import glob
import multiprocessing.pool
import os
import tarfile
import urllib.request
import warnings
import stat
import torch

from setuptools import setup, find_packages, distutils
from setuptools.command.install import install
from distutils import log
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, include_paths


def download_extract(url, dl_path):
    if not os.path.isfile(dl_path):
        # Already downloaded
        urllib.request.urlretrieve(url, dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()

FILES_TO_MARK_EXECUTABLE = ["flac-linux-x86", "flac-linux-x86_64", "flac-mac", "flac-win32.exe"]

# Does gcc compile with this header and library?
def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path \
              + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0


compile_args = ['-O3', '-DKENLM_MAX_ORDER=6', '-std=c++14', '-fPIC']
ext_libs = []
if compile_test('zlib.h', 'z'):
    compile_args.append('-DHAVE_ZLIB')
    ext_libs.append('z')

if compile_test('bzlib.h', 'bz2'):
    compile_args.append('-DHAVE_BZLIB')
    ext_libs.append('bz2')

if compile_test('lzma.h', 'lzma'):
    compile_args.append('-DHAVE_XZLIB')
    ext_libs.append('lzma')

third_party_libs = ["kenlm", "openfst-1.6.7/src/include", "ThreadPool", "boost_1_67_0", "utf8"]
compile_args.extend(['-DINCLUDE_KENLM', '-DKENLM_MAX_ORDER=6'])
lib_sources = glob.glob('third_party/kenlm/util/*.cc') + glob.glob('third_party/kenlm/lm/*.cc') + glob.glob(
    'third_party/kenlm/util/double-conversion/*.cc') + glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = ["D:\\SpeechRecognition\\AutomaticSpeechRecognitionSystem\\neuralnet", "D:\\SpeechRecognition\\AutomaticSpeechRecognitionSystem\\ctcdecode"]
ctc_sources = glob.glob('ctcdecode/src/*.cpp')

extension = CppExtension(
    name='ctcdecode._ext.ctc_decode',
    package=True,
    with_cuda=False,
    sources=ctc_sources + lib_sources,
    include_dirs=third_party_includes + include_paths(),
    libraries=ext_libs,
    extra_compile_args=compile_args,
    language='c++'
)


# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(os.cpu_count())
    list(thread_pool.imap(_single_compile, objects))
    return objects


# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile

class BuildExtension(install):
    def run(self):
        install.run(self)  # do the original install steps

        # mark the FLAC executables as executable by all users (this fixes occasional issues when file permissions get messed up)
        for output_path in self.get_outputs():
            if os.path.basename(output_path) in FILES_TO_MARK_EXECUTABLE:
                log.info("setting executable permissions on {}".format(output_path))
                stat_info = os.stat(output_path)
                os.chmod(
                    output_path,
                    stat_info.st_mode |
                    stat.S_IRUSR | stat.S_IXUSR |  # owner can read/execute
                    stat.S_IRGRP | stat.S_IXGRP |  # group can read/execute
                    stat.S_IROTH | stat.S_IXOTH  # everyone else can read/execute
                )

setup(
    name="CTC_SpeechRecognition",
    version="1.0.0",
    description="Automatic Speech Recognition System with PyTorch and CTC Decoder based on Paddle Paddle's implementation",
    url="https://github.com/DvelopedByIkshaan/CTC-SpeechRecognition",
    author="Ikshaan Gupta",
    author_email="ikshaan@gmail.com",
    # Exclude the build files.
    packages=["CTC_SpeechRecognition"],
    include_package_data=True,
    include_dirs=third_party_includes,
    cmdclass={"install": BuildExtension}
)