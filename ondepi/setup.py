import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from ondepi.settings import intensity_path, resources_path, home_path
import numpy


def scan_dir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scan_dir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        language="c++",
        include_dirs=['./', str(intensity_path), str(resources_path),
                      str(home_path), numpy.get_include()],
    )


extNames = scan_dir('resources')

extensions = [makeExtension(name) for name in extNames]

setup(
    name="ondepi",
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext': {'inplace': True, 'force': True}},
)

print('\n\n\n********CYTHON COMPLETE********')
