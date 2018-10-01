source ../venv/bin/activate
python model_performance.py
Rscript GenerateGraphics.R
echo "[BASH] Generating Graphs..."
cd Graphs
for i in $( ls ); do
    echo "[BASH] Working with model '$i'..."
    cd $i
    for d in $( echo */ ); do
        cd $d
        figure=${PWD##*/}
        echo -e "\r[BASH]\tAnimating '$figure'..."
        if [ ! -f $figure.gif ]; then
            convert -delay 10 -loop 0 * $figure.gif
        fi
        cd ..
    done
    cd ..
done
cd ..
echo "[BASH] Done!"
