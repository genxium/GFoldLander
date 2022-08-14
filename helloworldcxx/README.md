# Reference counting in Panda3D
Panda3D has a built-in [automatic reference counting mechanism](https://docs.panda3d.org/1.10/cpp/programming/object-management/reference-counting), where users can simply wrap the type of the declared variable by `PT()` to get a smart pointer for convenience.

A typical `subclass` of `class ReferenceCount` would be [PandaNode](https://www.panda3d.org/reference/cxx/classPandaNode.html).

# Useful Commands 
```bash
helloworldcxx> cd output 
helloworldcxx/output> rm CMakeCache.txt 
helloworldcxx/output> cmake ../ 
# [WARNING] or "cmake ../ -G Xcode" to generate an Xcode project 

# [WARNING] It's critical to specify "--config Release", see "CMakeLists.txt" for more information!
helloworldcxx/output> cmake --build . --config Release 
```
