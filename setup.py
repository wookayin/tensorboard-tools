from setuptools import setup

setup(
    name='tbtools',
    version='0.1.0',
    license='MIT',
    description='Tensorboard Tools',
    #long_description=read_readme(),
    url='https://github.com/wookayin/tensorboard-tools',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    keywords='tensorflow tensorboard',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    packages=['tbtools'],
    install_requires=[
        'psutil',
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'nose-cover3'],
    entry_points={
        'console_scripts': ['tb=tbtools.tb:main'],
    },
    include_package_data=True,
    zip_safe=False,
)
