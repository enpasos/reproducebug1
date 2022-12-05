# reproduce bug

Using a toy example derived from [djl mnist](https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_mnist_mlp.md).

## build

```
    gradlew build
```

## run

``` 
    java -jar app/build/libs/app-0.0.1-SNAPSHOT.jar  
```

## Example run

### Run1
**Result:**
- 4 MiB memory leak on GPU per epoch (looks constant)
- duration increase about 1 min per epoch (looks linear)

**Stack:**
- Java
  - DJL: 0.21.0-SNAPSHOT  (05.12.2022)
  - Java: Corretto-17.0.3.6.1
- PYTORCH: 1.12.1
- CUDA
  - CUDNN: cudnn-windows-x86_64-8.5.0.96_cuda11
  - CUDA SDK: 11.6.2
- OS
  - GPU Driver: 522.25
  - OS: Edition	Windows 11 Pro, Version	22H2, Betriebssystembuild	22623.1020
- HW
  - GPU: NVIDIA GeForce RTX 3090
  - CPU: AMD Ryzen 9 3950X 16-Core Processor
  - RAM: 64 GB


### Run2
**Result:**
- 4 MiB memory leak on GPU per epoch (looks constant)
- duration increase about 1 min (even a little more than on windows) per epoch (looks linear)

**Stack:**
- Java
    - DJL: 0.21.0-SNAPSHOT  (05.12.2022)
    - Java: Corretto-17.0.5.8.1
- PYTORCH: 1.12.1
- CUDA
    - CUDNN: cudnn-linux-x86_64-8.7.0.84_cuda11
    - CUDA SDK: 11.6.2
- **WSL**
    - **Ubuntu 20.04.4**
    - **WSL2**
- OS
    - GPU Driver: 522.25
    - OS: Edition	Windows 11 Pro, Version	22H2, Betriebssystembuild	22623.1020
- HW
    - GPU: NVIDIA GeForce RTX 3090
    - CPU: AMD Ryzen 9 3950X 16-Core Processor
    - RAM: 64 GB