# e-lab extra package for Torch7

This package provides extra functions for Torch7 from e-lab, Purdue University

## Install

To install this package, use the following command: 

```sh
$ luarocks make
```

## Short documentation

### Useful functions

#### ls - unix listing command

In order to add `eex.ls` to the global name space type the following Lua command

```lua
require 'eex'
ls = eex.ls
```

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

#### eex.datasetsPath - fetches $EEX_DATASETS environment variable

Returns the location of the *datasets* directory on the current machine. Basically it fetches the `$EEX_DATASETS` environment variable if existing, otherwise it prompts the user to input the location of the *datasets* dicrectory. In order to `ls` the content of the *abc-dataset* we can do as follow

```lua
ds = eex.datasetsPath()
print(ls(ds .. '/abc-dataset'))
```

### License

MIT
