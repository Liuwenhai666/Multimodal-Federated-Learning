source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir

mkdir -p SDWPF
cd SDWPF

# https://aistudio.baidu.com/competition/detail/152/0/introduction
wget https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a448e2c04a7499c43e1ae4cc28fbdee8e087e2385?responseContentDisposition=attachment%3B%20filename%3Dwtbdata_245days.csv&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-05T14%3A17%3A03Z%2F-1%2F%2F5932bfb6aa3af1bcfb467bf2a4a6877f8823fe96c6f4fd0d4a3caa722354e3ac

mv ./*.csv SDWPF.csv

# "https://uca47865c3f1427ee203887cd5fa.dl.dropboxusercontent.com/cd/0/get/CPdlBVeKiDrnRsMDVLLW82-KbwWkUshqVY8n8cr5BamFZoQBpp_PxLEDrYLMZPYl9MY96CAkzPWjPqiOotN9heR8rXY0Fan1U4xpWq0fEkMfxqdIdRC8ik2wQ-J37We2XoDj-Oki3VZvhmmZw5f52nAqQtOT3IdOY72qj4HDq0uneg/file?_download_id=849786806047999491602362772912384925430180091130411443931323519219&_notify_domain=www.dropbox.com&dl=1"