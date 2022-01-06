from setuptools import setup, find_packages

# Define version
__version__ = 0.4

setup(
    name="comptic",
    author="Zack Phillios",
    author_email="zack@zackphillips.com",
    version=__version__,
    description="Computational Microscopy Helper Functions",
    license="BSD",
    packages=find_packages(),
    include_package_data=True,
    py_modules=["comptic"],
    package_data={"": ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.json"]},
    install_requires=[
        "planar",
        "sympy",
        "numexpr",
        "contexttimer",
        "imageio",
        "matplotlib_scalebar",
        "tifffile",
        "numpy",
        "scipy",
        "scikit-image",
    ],
)
