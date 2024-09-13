## Access Server to Run the NoteBook on your terminal
```bash
ssh username@hostname.com
```
## Activate Conda Environment
```
cd ChangeDetection-Buddhi
source ~/.bashrc
conda activate cd
```
## Download Dataset
```
pip3 install gdown
gdown https://drive.google.com/uc?id=<file_id> -o SECOND.zip
unzip SECOND.zip
```
## Enable Background Runnning
```
tmux 
```
## Load Notebooks
```
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
# Copy the token here
```
## On WSL
```
ssh -N -f -L localhost:8888:hostname.com:8889 username@hostname2.com
```
## On your WebBrowser navigate to
```
http://localhost:8888
```
## If the connection terminated then:
```
tmux attach
```
