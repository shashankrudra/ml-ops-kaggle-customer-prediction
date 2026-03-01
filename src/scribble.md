

Make sure you have mlflow installed: pip install mlflow
This will log metrics, models, and the submission file to MLflow.
You can view results by running mlflow ui in your terminal and opening http://localhost:5000 in your browser.


pip uninstall xgboost -y
pip install --no-cache-dir xgboost

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade xgboost
brew upgrade libomp

git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow
git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask fetch --unshallow


pip uninstall xgboost
pip install xgboost

ls /usr/local/Cellar/libomp/

ls -l /usr/local/lib/libomp.dylib

brew cleanup libomp

pip uninstall xgboost
pip install --no-binary :all: xgboost


pip check
 
export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
  

this is execpted from code with lib symlink
ls -l /usr/local/lib/libomp.dylib

pip uninstall xgboost -y
pip install --no-cache-dir xgboost

Check Python: python -c "import platform; print(platform.machine())"

Check libomp: file /usr/local/opt/libomp/lib/libomp.dylib

export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"'
 

sudo ln -sf /usr/local/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install xgboost

 
pip uninstall xgboost -y
pip install --no-cache-dir xgboost

export OpenMP_ROOT=$(brew --prefix libomp)
pip install xgboost --no-binary xgboost

pip install --no-cache-dir xgboost


# 1. Clear out the old, broken version
pip uninstall xgboost -y

# 2. Tell the compiler where your Homebrew OpenMP is
export OpenMP_ROOT=$(brew --prefix libomp)

# 3. Perform the fresh, source-based install
pip install --no-cache-dir xgboost --no-binary xgboost


DYLD_PRINT_LIBRARIES=1 python src/test_xgboost_libomp.py


