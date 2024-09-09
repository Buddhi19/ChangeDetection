## Access Server to Run the NoteBook on your terminal
```bash
ssh username@hostname.com
```
## Activate Conda Environment
```
cd ChangeDetection-Buddhi
conda activate cd
```
## Download Dataset
```
pip3 install gdown
gdown https://drive.google.com/uc?id=<file_id> -o SECOND.zip
unzip SECOND.zip
```
## Load Notebooks
```python
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
