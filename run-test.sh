for f in in/*; do 
	python raw2vector.py $f --dump
done
