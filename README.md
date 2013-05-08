# e-lab extra package for Torch7

This package provides extra functions for Torch7 from e-lab, Purdue University

## Install

To install this package, use the following command: 

``` sh
$ luarocks make
```

## Short documentation

### Useful functions

#### ls - unix listing command

Listing the content of the current directory and fetching the number of elements it contains

```
ls()
#ls()
```

Listing the content of a specific directory `path` (`path` is a string)

```
ls(path)
```

Getting the *full file name* (file name preceded by its parent directories) of the 3rd PNG image contained in a specific directory `path`

```
ls(path .. '/*.png')[3]
```
