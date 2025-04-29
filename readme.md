Web crawl path is inside ir_project so please place it there
git clone https://github.com/Dk-21/IR_project.git
cd IR_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
python3 meta_data_loader.py
Python3 idf_builder.py
