#this file is used to create all the floders nd files using one file
#create a templets.sh file
#creating the directorys
mkdir -p src
mkdir -p researchs

#creating the files using in one floder .sh

touch src/__init__.py
touch src/helper.py
touch src/prompt.py

touch .env
touch setup.py
touch app.py
touch requriments.txt

touch researchs/trails.ipynb

echo "Directory and  files created successfully!."

#in git bash type sh templets.sh
#add this all floders into git add . , git commit -m " " , git push origin main

#for env 
#conda #create -n medical_chatbot python=3.10 -y
#conda activate medical_chatbot
#pip install -r requriments.txt

#now go to setup files