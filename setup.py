from distutils.core import setup

setup(
    name='mnfy',
    version='1.0.0',
    author='Brett Cannon',
    author_email='brett@python.org',
    url='http://mnfy.googlecode.com/',
    py_modules=['mnfy'],  # Don't install test code since not in a package
    license='Apache Licence 2.0',
    long_description=open('README').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.2'
    ],
)
