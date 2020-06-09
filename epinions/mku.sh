#!/bin/sh

trap `rm -f tmp.$$; exit 1` 1 2 15

for i in 1 2 3 4 5
do
	head -`expr $i \* 20000` epinions_new.data | tail -20000 > tmp.$$
	sort -t" " -k 1,1n -k 2,2n tmp.$$ > e$i.test
	head -`expr \( $i - 1 \) \* 20000` epinions_new.data > tmp.$$
	tail -`expr \( 5 - $i \) \* 20000` epinions_new.data >> tmp.$$
	sort -t" " -k 1,1n -k 2,2n tmp.$$ > e$i.base
done

./allbut.pl ea 1 10 100000 epinions_new.data
sort -t" " -k 1,1n -k 2,2n ea.base > tmp.$$
mv tmp.$$ ea.base
sort -t" " -k 1,1n -k 2,2n ea.test > tmp.$$
mv tmp.$$ ea.test

./allbut.pl eb 11 20 100000 epinions_new.data
sort -t" " -k 1,1n -k 2,2n eb.base > tmp.$$
mv tmp.$$ eb.base
sort -t" " -k 1,1n -k 2,2n eb.test > tmp.$$
mv tmp.$$ eb.test

