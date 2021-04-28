# Installing relevant packages

To install cv2, datawig and fancyimpute, do following:


1. Log in to terminal
2. `module purge`
3. Create and activate a virtual env:
```bash 
python -m venv hack_env
source hack_env/bin/activate
```
4. Proceed with following:
```bash 

pip install --upgrade pip

pip install opencv-python
pip install datawig
pip install fancyimpute

```
5. For testing, execute `helpers/test_load_packages.py` from the repo
