include $(top_srcdir)/Common.mk


export PYTHON_PACKAGES
export PYTHONPATH
export PYTHONUSERBASE










PYTHON_PACKAGES = six numpy wheel mock
# tensorflow-model-optimization

PYTHONPATH = $(abs_top_builddir)/conf/python/site-packages/lib/python3.6/site-packages/:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3.6/site-packages/:$(abs_srcdir)

##    .########.########.##....##..######...#######..########..########.##........#######..##......##  
##    ....##....##.......###...##.##....##.##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....##.......####..##.##.......##.....##.##.....##.##.......##.......##.....##.##..##..##  
##    ....##....######...##.##.##..######..##.....##.########..######...##.......##.....##.##..##..##  
##    ....##....##.......##..####.......##.##.....##.##...##...##.......##.......##.....##.##..##..##  
##    ....##....##.......##...###.##....##.##.....##.##....##..##.......##.......##.....##.##..##..##  
##    ....##....########.##....##..######...#######..##.....##.##.......########..#######...###..###.  

DOWNLOADS = tensorflow_src
tensorflow_src: ##@tf download git tensorflow code (master)
tensorflow_src_URL    = https://github.com/tensorflow/tensorflow.git
tensorflow_src_BRANCH = master

DOCKER_TARGETS = tf-%
tf-%: DOCKER_CONTAINER = tf-build

#IMAGE: tensorflow/tensorflow:nightly-devel-py3
tf-init: ##@tf build process
tf-init: DOCKER_IMAGE = tf-devel-py3
tf-init: DOCKER_URL = $(srcdir)/docker
tf-init: DOCKER_MOUNTS = dist-packages:/usr/local/lib/python3.6/dist-packages /run/user:/run/user
tf-init: DOCKER_PORTS = 8888:8888 6006:6006 3000:3000
tf-init: pip-install | tensorflow_src
	@ :; 

# MDSplus setup to access Analisys variables (offline)
# RUN_PREQUEL = treeSetSource A inProgress;

PYTHON_PACKAGES += bokeh bokeh-metaplot sklearn

# SPARSIFIER
# tensorflow-model-optimization

# BREVITAS
# click  onnx  


DOCKER_TARGETS += pip-install
pip-install: DOCKER_CONTAINER = tf-build


tensorflow_src/.tf_configure.bazelrc: | tf-init
	@ cd tensorflow_src; ./configure

tf-configure: ##@tf reconfigure src code project
tf-configure: tensorflow_src/.tf_configure.bazelrc

tf-build: ##@tf build tensorflow code (not needed)
tf-build: tf-configure | tf-init
	@ cd tensorflow_src; bazel build --config=opt //tensorflow_src/tools/pip_package:build_pip_package

tf-shell: ##@tf enter container with shell (bash)
tf-shell: tf-init
	@ $(RUN_PREQUEL) bash -l

## ...##.##.....##.##.........###....########..########.
## ...##.##.....##.##........##.##...##.....##.##.....##
## .#########.#########.....##...##..##.....##.##.....##
## ...##.##.....##.##......##.....##.########..########.
## .#########.#########....#########.##........##.......
## ...##.##.....##.##......##.....##.##........##.......
## ...##.##.....##.##......##.....##.##........##.......

export abs_srcdir
export abs_builddir
export rfx_ver = inProgress

# automatically reload the following files (this works only if 'ipysh' has been imported too)
export IPY_AIMPORT = Hunch_utils \
					 Dummy_g1data \
					 Hunch_lsplot \
					 Hunch_tSNEplot


DOCKER_TARGETS += te_db_1.npy
te_db_1.npy: ##@data create npy database 
te_db_1.npy: DOCKER_CONTAINER = tf-build
te_db_1.npy: create_db_1.py | pip-install
	@ python $< $(ARGS)


DOCKER_TARGETS += runpy
PYSCRIPTS = examples/AEFIT_qsh.py
runpy: ##@jp Run ipython console starting by script defined in $NAME or $PYSCRIPTS
runpy: DOCKER_CONTAINER = tf-build
runpy: $(NAME) $(PYSCRIPTS) te_db_1.npy | pip-install
	@ $(info RUNNING: $<)
	  $(RUN_PREQUEL) ipython -i $< $(ARGS)

DOCKER_TARGETS += jupyter
jupyter: DOCKER_CONTAINER = tf-build
jupyter: ##@jp start jupyter server on jpnb $(srcdir) folder
	@ $(RUN_PREQUEL) jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=$(abs_srcdir)/jpnb

DOCKER_TARGETS += jupyter-lab
jupyter-lab: DOCKER_CONTAINER = tf-build
jupyter-lab: ##@jp start jupyter server on jpnb $(srcdir) folder
	@ $(RUN_PREQUEL) jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=$(abs_srcdir)/jpnb


jupyter-theme: ##@jp setup theme
	jt -t onedork -cellw 95\% -T


DOWNLOADS += rdipietro
rdipietro: ##download rdpietro notebooks
rdipietro_URL = https://github.com/rdipietro/jupyter-notebooks.git
rdipietro_BRANCH = master
rdipietro_DIR = $(srcdir)/jpnb/rdipietro




# TORCH_LIB_PATH = 	$(abs_top_builddir)/ext/pytorch/build/lib \
# 					$(abs_top_builddir)/ext/pytorch/build/lib.linux-x86_64-3.7

# TORCH_PY_PATH = $(abs_top_builddir)/ext/pytorch/build/lib.linux-x86_64-3.7 \
# 				$(abs_top_builddir)/ext/pytorch

# ONNX_LIB_PATH = $(TORCH_LIB_PATH)
# ONNX_PY_PATH = 	$(abs_top_builddir)/ext/onnx/build/lib.linux-x86_64-3.7 \
# 				$(abs_top_builddir)/ext/onnx \
# 				$(TORCH_PY_PATH)


# FINN_WS = $(abs_top_builddir)/ext/FINN
# FINN_LIB_PATH =  $(ONNX_LIB_PATH)
# FINN_PY_PATH = 	\
# 					$(FINN_WS)/src \
# 					$(FINN_WS)/brevitas_cnv_lfc/training_scripts \
# 					$(FINN_WS)/brevitas \
# 					$(FINN_WS)/pyverilator \
# 					$(FINN_WS)/notebooks \
# 					$(ONNX_PY_PATH)
					

# FINN_INST_NAME = finn_test
# DIRECTORIES = /tmp/$(FINN_INST_NAME)

# export PYTHONDONTWRITEBYTECODE = 1
# export FINN_INST_NAME := $(FINN_INST_NAME)
# export FINN_WS := $(FINN_WS)
# export VIVADO_PATH := $(or $(VIVADO_PATH),/mnt/data0/Xilinx/Vivado/2019.1/)
# export PYNQSHELL_PATH := $(FINN_WS)/PYNQ-HelloWorld/boards
# export FINN_ROOT := $(FINN_WS)
# export LD_LIBRARY_PATH := $(call ak__join_with,:,$(FINN_LIB_PATH))$(if ${LD_LIBRARY_PATH},:${LD_LIBRARY_PATH})
# export PYTHONPATH := $(call ak__join_with,:,$(FINN_PY_PATH))$(if ${PYTHONPATH},:${PYTHONPATH})





# .########.########....##..
# ....##....##........####..
# ....##....##..........##..
# ....##....######......##..
# ....##....##..........##..
# ....##....##..........##..
# ....##....##........######

DOCKER_TARGETS += tf1-%
tf1-%: DOCKER_CONTAINER = tf1-stable


#IMAGE: tensorflow/tensorflow:nightly-devel-py3
tf1-init: ##@tf build process
tf1-init: DOCKER_IMAGE = tf-stable-py3
tf1-init: DOCKER_URL = $(srcdir)/docker_v1
tf1-init: DOCKER_MOUNTS = dist-packages-v1:/usr/local/lib/python3.5/dist-packages
# tf1-init: DOCKER_PORTS = 8888
tf1-init: 
	@ :; 

tf1-shell: ##@tf enter container with shell (bash)
tf1-shell: tf1-init
	@ bash -l


runpy_v1 tf1-%: PYTHONPATH = $(abs_srcdir)

DOCKER_TARGETS += runpy_v1
runpy_v1: ##@jp Run ipython console starting by script defined in $NAME or $PYSCRIPTS
runpy_v1: DOCKER_CONTAINER = tf1-stable
runpy_v1: $(NAME) examples/AEFIT_dummy_v1.py te_db_1.npy | pip-install
	@ ipython3 -i $< $(ARGS)

#
# IF pip does not work run:   python -m pip install --upgrade --force-reinstall pip
#



##   .########..########.########..########..########..######.....###....########.########.########.
##   .##.....##.##.......##.....##.##.....##.##.......##....##...##.##......##....##.......##.....##
##   .##.....##.##.......##.....##.##.....##.##.......##........##...##.....##....##.......##.....##
##   .##.....##.######...########..########..######...##.......##.....##....##....######...##.....##
##   .##.....##.##.......##........##...##...##.......##.......#########....##....##.......##.....##
##   .##.....##.##.......##........##....##..##.......##....##.##.....##....##....##.......##.....##
##   .########..########.##........##.....##.########..######..##.....##....##....########.########.

# tf-install-vscode: USER=root
# tf-install-vscode: ##@@tf install vscode inside docker container (not working in docker-targets)
# tf-install-vscode:
# 	@ \
# 	 apt-get update; \
# 	 apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
# 	 apt install -y libasound2 git; \
# 	 apt install -y python3-tk pylint; \
# 	 wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -; \
# 	 add-apt-repository -ru "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
# 	 add-apt-repository -u "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
# 	 apt install -y code;


# tf-install-mdsplus: export PYTHONUSERBASE=/usr/local/lib/python3.6/dist-packages/
# tf-install-mdsplus: USER=root
# tf-install-mdsplus: ##@tf install mdsplus in container
# tf-install-mdsplus:
# 	@ \
# 	  apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
# 	  wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
# 	  apt-key add mdsplus.gpg.key; \
# 	  add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
# 	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus stable"; \
# 	  mkdir -p /usr/lib/python3.6/site-packages/; \
# 	  apt install -y mdsplus mdsplus-python;


# tf-install-mdsplus-alpha: USER=root
# tf-install-mdsplus-alpha: ##@tf install mdsplus-alpha in container
# tf-install-mdsplus-alpha:
# 	@ \
# 	  apt install -y gpg-agent software-properties-common apt-transport-https git wget libasound2 pylint; \
# 	  wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
# 	  apt-key add mdsplus.gpg.key; \
# 	  add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus alpha"; \
# 	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus alpha"; \
# 	  mkdir -p /usr/lib/python3.6/site-packages/; \
# 	  apt install -y mdsplus-alpha mdsplus-alpha-python;



# DOCKER_TARGETS += code-server-%
# CODE_SERVER_PORT = 8080

# code-server-init: DOCKER_PORTS = $(CODE_SERVER_PORT):$(CODE_SERVER_PORT)
# code-server-init: DOCKER_IMAGE = codercom/code-server:latest
# code-server-init: DOCKER_CONTAINER = code-server
# code-server-init:
# 	:;

# code-server-start: ##@@ide start vs code server via docker container
# code-server-start: code-server-init
# code-server-start: DOCKER_CONTAINER = code-server
# code-server-start: 
# 	dumb-init -- code-server --host 0.0.0.0 --port $(CODE_SERVER_PORT) --auth none $(top_srcdir)

# code-server-stop: ##@@ide stop vs code server via docker container
# code-server-stop: DOCKER_CONTAINER = code-server
# code-server-stop: docker-clean




