import os

from setuptools import find_packages
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy

# get the numpy include dir
numpy_include_dir = numpy.get_include()

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'rndf_robot.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'src/rndf_robot/utils/libkdtree/pykdtree/kdtree.c',
        'src/rndf_robot/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy.get_include()],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'rndf_robot.utils.libmcubes.mcubes',
    sources=[
        'src/rndf_robot/utils/libmcubes/mcubes.pyx',
        'src/rndf_robot/utils/libmcubes/pywrapper.cpp',
        'src/rndf_robot/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)


# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'rndf_robot.utils.mesh_util.triangle_hash',
    sources=[
	'src/rndf_robot/utils/mesh_util/triangle_hash.pyx',
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir],
    language='c++'
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'rndf_robot.utils.libmise.mise',
    sources=[
        'src/rndf_robot/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'rndf_robot.utils.libsimplify.simplify_mesh',
    sources=[
        'src/rndf_robot/utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'rndf_robot.utils.libvoxelize.voxelize',
    sources=[
        'src/rndf_robot/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


packages = find_packages('src')
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'ndf_robot' or p.startswith('ndf_robot.')


def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths


extra_pkg_files = pkg_files('src/ndf_robot/descriptions')

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    name='ndf_robot',
    author='Anthony Simeonov, Yilun Du',
    license='MIT',
    packages=packages,
    package_dir={'': 'src'},
    package_data={
        'ndf_robot': extra_pkg_files,
    },
    install_requires=read_requirements_file('requirements.txt'),
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
        # 'build_ext': BuildExtension
    }
)

