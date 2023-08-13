# neuroNet
Experimental code to learn building a neural network from scratch.

## Installing Cuda for Ubuntu 22.04 LTS in WSL

```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

Then I did a
```
$ sudo update-pciids
$ lspci | grep nvidia
```

An nVidia device had to be shown, but it didn't. However, it turned out later that the samples were running fine on my device.

Then I did
```
$ sudo apt-key del 7fa2af80
```
Not clear what this did and if it did anything in the first place.

Adding the cuda binary to the `PATH` variable:
```
$ export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
```
But I expect that this needs to be done automatically.


Download the Cuda samples:
```
$ git clone https://github.com/NVIDIA/cuda-samples.git
```


```
$ cd cuda-samples/
$ cd Samples/
$ cd 1_Utilities/
$ cd bandwidthTest/
$ make
$ ./bandwidthTest

$ cd ..

$ cd deviceQuery
$ make
$ ./deviceQuery
```

Cleaning up
```
$ sudo apt-get remove --purge "cuda-repo-<distro>-X-Y-local*"
```
