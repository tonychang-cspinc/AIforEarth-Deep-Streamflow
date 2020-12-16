FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

LABEL org.label-schema.license="CSP-1.0" \
    org.label-schema.vcs-url="https://gitlab.com/tonychang" \
    org.label-schema.vendor="Conservation Science Partners" \
    maintainer="Tony Chang <tony@csp-inc.org>"
   
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
    apt-utils \
    ed \
    less \
    locales \
    vim-tiny \
    wget \
    ca-certificates \
    apt-transport-https \
    software-properties-common \
    gcc \
    g++ \
    git \
    gfortran \
    gsfonts \
    gnupg2 \
    graphviz \
    imagemagick \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    liblzma-dev \
    liblzma-dev \
    libpcre3-dev \
    libpcre2-dev \
    libbz2-dev \
    libblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    libicu-dev \
    libcurl4-openssl-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libssl-dev \
    libxml2-dev\
    make \
    nodejs \
    pkg-config \
    unzip \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install jupyterlab \
    pandas \
    numpy \
    pyyaml \
    pydot \
    Pillow \
    scipy \
    scikit-learn \
    scikit-image \
    seaborn \
    symbolic \
    geopandas \
    graphviz \
    descartes \
    rasterio \
    folium \
    pydot \
    tensorflow-probability \
    tensorflow-addons \
    tfds-nightly \
    tqdm \
    bokeh \
    opencv-python \ 
    pydensecrf \
    keras \
    keras-ocr \
    gpsphoto \
    exifread \
    piexif \
    jug \
    xlrd 

#add vim bindings
RUN mkdir -p $(jupyter --data-dir)/nbextensions &&\
	 cd $(jupyter --data-dir)/nbextensions &&\
	 git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding &&\
	 jupyter nbextension enable vim_binding/vim_binding
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager &&\
#    jupyter labextension install @bokeh/jupyter_bokeh
