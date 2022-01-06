# comptic: Computational Microscopy Support library
Maintained by Zack Phllips

## Requirements
numpy, scipy, jac

## Installation
The easy way:
```shell
pip install https://www.github.com/zack-insitro/comptic
```

The developer way:
```shell
git clone https://www.github.com/zack-insitro/comptic
cd comptic
python setup.py develop
```

## Module List
- ***comptic.camera***: Camera-related tools for demosaicing, etc.
- ***comptic.containers***: Data structures for datasets and metadata
- ***comptic.error***: Error metric functions
- ***comptic.imaging***: Imaging related functions (such as pupil and OTF generation)
- ***comptic.ledarray***: Tools for generating LED array positions
- ***comptic.noise***: Tools for adding and measuring noise in images
- ***comptic.prop***: Tools for optical propagation
- ***comptic.ransac***: RANSAC implementations
- ***comptic.registration***: Tools for image registration
- ***comptic.simulation***: Tools for simulating objects
- ***comptic.transformation***: Tools for applying transformations to Images
- ***comptic.dpc***: Tools for DPC inversion
- ***comptic.fpm***: Tools for Fourier Ptychography

## License
BSD 3-Clause
