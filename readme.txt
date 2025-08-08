uvicorn accountslogin.main:app

Ensure you have the required libraries for pillow:

Install zlib and other necessary libraries.
On Windows, you can use a package manager like choco (Chocolatey) to install these dependencies:
choco install zlib
choco install libjpeg-turbo

c:\Python312\python.exe -m pip install 
c:\Python312\python.exe -m pip install langchain
c:\Python312\python.exe -m pip install chromadb
c:\Python312\python.exe -m pip install pypdf
c:\Python312\python.exe -m pip install pytest

activate your enviroment before you call install command.


pip install -r requirements.txt
c:\Python312\python.exe -m pip uninstall -r requirements.txt
c:\Python312\python.exe -m pip install -r requirements.txt

c:\Python312\python.exe -m pip install jinja2
c:\Python312\python.exe -m pip install "unstructured[md]"
c:\Python312\python.exe -m pip install "unstructured[pdf]"
Selecting the Python Interpreter (Activated your Python environment)
Open the Command Palette by pressing Ctrl+Shift+P.
Type Python: Select Interpreter and press Enter.

Select the python directory folder where the fastapi library installed into.
Python 3.10.8 64-bit ~\AppData\Local\Programs\Python\Python310\python.exe

Python 3.12.3 64-bit c:\Python312\python.exe



您可以使用 pip 列出特定庫（如 jwt）的可用版本。以下是您可以使用的命令：

列出 jwt 庫的可用版本
使用 pip 搜索命令
另一種方法是使用 pip search 命令來搜索並列出 jwt 庫的版本：

sh
Copy code
pip index versions PyJWT



Show library version
c:\Python312\python.exe -m pip show PyJWT

c:\Python312\python.exe -m pip uninstall PyJWT

c:\Python312\python.exe -m pip install PyJWT==2.8.0

c:\Python312\python.exe -m pip show starlette

c:\Python312\python.exe -m pip uninstall starlette

c:\Python312\python.exe -m pip install starlette==0.37.2


c:\Python312\python.exe -m pip install  gTTS==2.5.1

c:\Python312\python.exe -m pip install pdfplumber markdownify

c:\Python312\python.exe -m pip show pdfplumber markdownify
=============Adding page=================

you need to create {page_name}.html, {page_name_content}.html, and {PageName}Controller.py
and then create /resources/static/css/{page_name}.css
and then create /resources/static/js/{page_name}.js
and then create /resources/static/js/app.controllers.ui.{page_name}.js
and then create /resources/static/js/{page_name}.js
and then update the /controllers/_init_.py


==================normal intallation==========================
# Navigate to your project directory
cd C:\WebDevelopment\Project\python-workspace\python-webapp\LearnWiseAI\LearnWiseLocalAI\sandbox

# Create a virtual environment
python -m venv .venv

pip install uvicorn

activate a virtual environment:
source venv/bin/activate  # On Windows use `.venv\Scripts\activate`
python.exe -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate


==================== pytest environment setting and using========================


在使用 `requirements.txt` 安裝 Python 套件之前，我們需要先建立並啟動虛擬環境。這樣可以確保我們安裝的套件不會和系統的其他套件產生衝突，並且可以更方便地管理和隔離項目的依賴。

以下是建立並啟動虛擬環境的步驟：

1. **建立虛擬環境**：
   首先，我們需要在項目目錄中建立一個虛擬環境。這可以通過以下指令來完成：
   ```bash
   python -m venv .venv
   ```
   上述指令會在當前目錄下建立一個名為 `.venv` 的虛擬環境。

2. **啟動虛擬環境**：
   建立虛擬環境後，我們需要啟動它。啟動虛擬環境的方法根據操作系統的不同而有所不同：

   - **在 Windows 上**：
     ```bash
     .venv\Scripts\activate
     ```

   - **在 macOS 和 Linux 上**：
     ```bash
     source .venv/bin/activate
     ```

   啟動虛擬環境後，命令提示符會顯示虛擬環境的名稱（例如 `(venv)`），這表示我們已經進入了虛擬環境。

3. **安裝 `requirements.txt` 中的套件**：
   當虛擬環境啟動後，我們可以使用 `pip` 安裝 `requirements.txt` 中列出的所有套件：
   ```bash
   pip install -r requirements.txt

   pip install -r accountslogin/requirements.txt
   ```

這樣做的好處是所有的套件會安裝在虛擬環境中，不會影響系統的全局 Python 環境。並且當我們完成項目後，只需刪除虛擬環境目錄即可完全移除這些依賴。

總結：
1. 建立虛擬環境：`python -m venv .venv`
2. 啟動虛擬環境：
   - Windows：`.venv\Scripts\activate`
   - macOS/Linux：`source venv/bin/activate`
3. 安裝套件：`pip install -r requirements.txt`

希望這些步驟對你有所幫助！


pip install google-cloud-speech

========================================================================
echo "# FastWhisper" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/fusumwan/FastWhisper.git
git push -u origin main

======================================================================




git remote add origin https://github.com/fusumwan/FastWhisper.git
git branch -M main
git push -u origin main


=================================

git pull
git add .
git commit -m "Updating"
git branch -M main
git push -u origin main



===============================For Windows==========================================
python -m venv .venv
.venv\Scripts\activate

====linux / mac=====
source venv/bin/activate
================
pip install --upgrade pip
pip install -r requirements.txt


# Export requirements.txt
pip freeze > requirements.txt


=======================================
===============================For Mac==========================================
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


# Export requirements.txt
pip freeze > requirements.txt




