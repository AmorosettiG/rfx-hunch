# RFX-Hunch

**Characterization of a new deep learning approach** [1] **for data recovery in the Soft X-Ray fusion plasma diagnostics** [2] **in RFX-mod**

- [1] A. Rigoni Garola et al., "Diagnostic Data Integration Using Deep Neural Networks for Real-Time Plasma Analysis," in IEEE Transactions on Nuclear Science, vol. 68, no. 8, pp. 2165-2172, Aug. 2021, https://doi.org/10.1109/TNS.2021.3096837
- [2] Franz, P., Gobbin, M., Marrelli, L., Ruzzon, A., Bonomo, F., Fassina, A., Martines, E., & Spizzo, G. (2013, April 23). Experimental investigation of electron temperature dynamics of helical states in the RFX-Mod reversed field pinch. Nuclear Fusion, 53(5), 053011. https://doi.org/10.1088/0029-5515/53/5/053011

-----------------------------------------------------------------

further documentation is available in [USER MANUAL](doc/README.md)



This is a autoconf/automake minimal setup to start a new project. It uses Kconfig integration 
provided by https://github.com/AndreaRigoni/autoconf-kconfig as a submodule in conf/kconfig

In the following a easy setup procedure:

<pre>
git clone https://github.com/andrearigoni/autoconf-bootstrap.git
cd autoconf-bootstrap
./bootstrap
mkdir build
cd build
../configure --enable-kconfig
# enjoy
</pre>

