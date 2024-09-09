## Access Server to Run the NoteBook on your terminal
```bash
ssh username@hostname.com
```
## Activate Conda Environment
```
cd ChangeDetection-Buddhi
conda activate cd
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