# a statistical tool to analyze the "scaling property" in Chinese courpus
Version: 2018_April_18
---------
###Counting collocations for each character 
###Counting links for each word 
---------
          
 * a funciton counting the number of collocations in a script, adding a new column called "#collocations" to the data frame char, and saving the result to the column.
 * download count_col.py to your desktop, save it to the same folder where the .ipynb files exist. 
 * see fake_generate_LogNormal_crazy_version.ipynb for examples.
 * download count_col.py to your desktop, save it to the same folder where the .ipynb files exist.
 * don't forget to put 
 'from count_col import count_col'
 to the top of your code!
 
count_links(word[,feaure])
  * input: word (pandas data frames)
  * a funciton counting how many links in a script, adding a new column called "#links" to the data frame word, and saving the result to that column.
  * count_links has been added to count.py file.
  * see count_link_frog.ipynb for examples.

 
