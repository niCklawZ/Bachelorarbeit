# Bachelorarbeit
Dieses Projekt enthält den Entwicklungsprozess und die finale Version der Software, die im Rahmen der Bachelorarbeit von Nick Kottek erstellt wurde. Es wurde eine Software erstellt, welche die Gesten des deutschen Fingeralphabets mithilfe eines Convolutional Neural Network klassifizieren kann.

# Installationsanweisungen
[Wichtiger Guide von Tensorflow](https://www.tensorflow.org/install/pip#windows-wsl2)

## Python
[Download 3.9](https://www.python.org/downloads/release/python-390/)

## Nvidia GPU Treiber
[Download Game Ready Driver](https://www.nvidia.com/download/index.aspx)

## Visual Studio Code
Wird bei der Installation vom CUDA Toolkit gefordert. Dort wird auch ein Link bereitgestellt.  
Installation des CUDA Toolkits abbrechen, VSCode installieren und danach CUDA Installation erneut starten.

## CUDA Toolkit
[Download v11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

## cuDNN SDK
[Download v8.6.0 für CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)  
[Schritte 1-4 ausführen](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows)

## WSL
[Step 2.2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)  
[Erste Schritte bis einschließlich Update- und Upgradepakete ausführen](https://learn.microsoft.com/de-de/windows/wsl/setup/environment#get-started)

- In dem Ordner des Projekts eine CMD öffnen.  
- `wsl` eingeben, um die WSL Umgebung zu starten  
- `nvidia-smi` eingeben. Das sollte eine Ausgabe erzeugen.  
- `sudo apt install python3-pip` pip installieren
- `pip install --upgrade pip` pip updaten
- `python3 -m pip install tensorflow[and-cuda]` um tensorflow mit GPU Support zu installieren.
- Mit `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` herausfinden,
ob die Installation geklappt hat und eine GPU gefunden wird. 

## Notwendige Packages installieren

Ich weiß nicht mehr alle, die benötigt wurden, aber hier ein paar:

- mediapipe (Hatte version 0.10.9)
- opencv `sudo apt-get update && sudo apt-get install -y python3-opencv` und `pip install opencv-python` 
(Aufjedenfall nicht opencv-python-headless installieren, damit gab es Probleme)
- Bei Problemen bezüglich `Qt platform plugin`: `sudo apt-get install python3-pyqt5` und `sudo apt-get install libxcb-xinerama0` [Quelle](https://github.com/labelmeai/labelme/issues/842)

Hier ist eine Liste der installierten Packages, bei einem Stand, wo alles wie gewollt funktioniert.
Die Liste kann per `pip list` ausgegeben werden.

<details>
    <summary>Package List</summary>

    absl-py                      2.1.0  
    astunparse                   1.6.3  
    attrs                        23.2.0  
    blinker                      1.4  
    cachetools                   5.3.3  
    certifi                      2024.2.2  
    cffi                         1.16.0  
    charset-normalizer           3.3.2  
    command-not-found            0.3  
    contourpy                    1.2.0  
    cryptography                 3.4.8  
    cycler                       0.12.1  
    dbus-python                  1.2.18  
    distro                       1.7.0  
    distro-info                  1.1+ubuntu0.2  
    flatbuffers                  23.5.26  
    fonttools                    4.49.0  
    gast                         0.5.4  
    google-auth                  2.28.1  
    google-auth-oauthlib         1.2.0  
    google-pasta                 0.2.0  
    grpcio                       1.62.0  
    h5py                         3.10.0  
    httplib2                     0.20.2  
    idna                         3.6  
    importlib-metadata           4.6.4  
    jeepney                      0.7.1  
    keras                        2.15.0  
    keyring                      23.5.0  
    kiwisolver                   1.4.5  
    launchpadlib                 1.10.16  
    lazr.restfulclient           0.14.4  
    lazr.uri                     1.0.6  
    libclang                     16.0.6  
    Markdown                     3.5.2  
    MarkupSafe                   2.1.5  
    matplotlib                   3.8.3  
    mediapipe                    0.10.9  
    ml-dtypes                    0.2.0  
    more-itertools               8.10.0  
    netifaces                    0.11.0  
    numpy                        1.26.4  
    nvidia-cublas-cu12           12.2.5.6  
    nvidia-cuda-cupti-cu12       12.2.142  
    nvidia-cuda-nvcc-cu12        12.2.140  
    nvidia-cuda-nvrtc-cu12       12.2.140  
    nvidia-cuda-runtime-cu12     12.2.140  
    nvidia-cudnn-cu12            8.9.4.25  
    nvidia-cufft-cu12            11.0.8.103  
    nvidia-curand-cu12           10.3.3.141  
    nvidia-cusolver-cu12         11.5.2.141  
    nvidia-cusparse-cu12         12.1.2.141  
    nvidia-nccl-cu12             2.16.5  
    nvidia-nvjitlink-cu12        12.2.140  
    oauthlib                     3.2.0  
    opencv-contrib-python        4.9.0.80  
    opencv-pyton                 4.9.0.80
    opt-einsum                   3.3.0  
    packaging                    23.2  
    pillow                       10.2.0  
    pip                          23.3.2  
    protobuf                     3.20.3  
    pyasn1                       0.5.1  
    pyasn1-modules               0.3.0  
    pycparser                    2.21  
    PyGObject                    3.42.1  
    PyJWT                        2.3.0  
    pyparsing                    2.4.7  
    PyQt5                        5.15.6
    PyQt5-sip                    12.9.1
    python-apt                   2.4.0+ubuntu3  
    python-dateutil              2.8.2  
    PyYAML                       5.4.1  
    requests                     2.31.0  
    requests-oauthlib            1.3.1  
    rsa                          4.9  
    SecretStorage                3.3.1  
    setuptools                   59.6.0  
    six                          1.16.0  
    sounddevice                  0.4.6  
    systemd-python               234  
    tensorboard                  2.15.1  
    tensorboard-data-server      0.7.2  
    tensorflow                   2.15.0.post1  
    tensorflow-estimator         2.15.0  
    tensorflow-io-gcs-filesystem 0.36.0  
    termcolor                    2.4.0  
    typing_extensions            4.10.0  
    ubuntu-advantage-tools       8001  
    ufw                          0.36.1  
    unattended-upgrades          0.1  
    urllib3                      2.2.1  
    wadllib                      1.3.6  
    Werkzeug                     3.0.1  
    wheel                        0.37.1  
    wrapt                        1.14.1  
    zipp                         1.0.0
</details>

## Ausführen der Skripts
`python3 <name>.py` eingeben, um das jeweilige Skript auszuführen.

**Hinweis:** Gegebenenfalls ist es notwendig im Code etwas anzupassen. 
In der `prediction.py` muss bspw. evtl. die URL zu Kamera Quelle angepasst werden.

## Troubleshooting: Links die mir irgendwann mal geholfen haben

- https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
- https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

