```
helloworldcxx> cd output 
helloworldcxx/output> rm CMakeCache.txt 
helloworldcxx/output> cmake ../ 
# [WARNING] or "cmake ../ -G Xcode" to generate an Xcode project 

# [WARNING] It's critical to specify "--config Release", see "CMakeLists.txt" for more information!
helloworldcxx/output> cmake --build . --config Release 
```
