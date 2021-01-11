echo "preprocess/readmission_dice_code/bin/desc -inputfiles $1 -outputfolder $2" >> $GEMINI_HOME/inference_logfile
preprocess/readmission_dice_code/bin/desc -inputfiles $1 -outputfolder $2
echo "preprocess/readmission_dice_code/bin/completer_inference -inputfiles $1 -inputfolder $2 -outputfolder $2 -visfolder $3" >> $GEMINI_HOME/inference_logfile
preprocess/readmission_dice_code/bin/completer_inference -inputfiles $1 -inputfolder $2 -outputfolder $2 -visfolder $3
