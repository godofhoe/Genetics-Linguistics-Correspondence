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

--------- 
<br>
<font size = 15>Fitting Zipf's law (discrete):</font>
<br><br>
The discrete Zipf's distribution is : <br>
\begin{equation}P_k=Ck^{-\alpha}\end{equation}
The normalized condition gives:<br>
\begin{equation}1=\sum P_k=C\sum_{m}^{n}(\frac{1}{k^\alpha}) = C(\zeta(\alpha, m) - \zeta(\alpha, n+1))\Rightarrow C=1/H(m,n,\alpha)\end{equation}
Where $H(m,n,\alpha)\equiv \sum_{m}^{n}\frac{1}{k^\alpha}=\zeta(\alpha, m) - \zeta(\alpha, n+1)$, $\zeta(\alpha, m)$ is Hurwitz zeta function.<br>

Now give a raw data $X=[x_1,x_2,x_3...,x_N]$ where $x_i$ is the word of utterance, the likelihood is:<br>
\begin{equation}L=\prod_{i=1}^{N}P(x_i|\alpha)=\prod_{i=1}^{k}P(y_i|\alpha)^{\rho(y_i)}=\prod_{i=1}^{k}\big[\frac{y_i^{-\alpha}}{H(y_1,y_k,\alpha)}\big]^{\rho(y_i)}\end{equation}
where $Y=[y_1, y_2...,y_k]$ is a rank set of non-repeat $X$ with frequency $\rho(y_i)$


The maximum likelihood estimation (MLE) requires $L$ be maximum, and then $\ln(L)$ will also be max:<br>
\begin{equation}0=\frac{\partial\ln{L}}{\partial \alpha}=-\sum_{i=1}^{k}{\rho(y_i)\ln{y_i}}+\frac{N}{H(y_1,y_k,\alpha)}\sum_{i=1}^{k}\frac{\ln{(y_i)}}{y_i^{ \alpha}}\end{equation}

However we can't solve $\alpha$ exactly in this form. Instead, we use the minimize function in scipy:
\begin{equation}max(\ln L(\alpha))=min(-1*\ln L(\alpha))\end{equation}
\begin{equation}\Rightarrow \alpha = \alpha_1, C=1/H(y_1,y_k,\alpha_1)\end{equation}




\begin{equation}\end{equation}

\begin{equation}\end{equation}


<ref>Reference: <br>
1. https://arxiv.org/pdf/cond-mat/0412004.pdf Appendix.A
2. scipy.minimize: https://www.youtube.com/watch?v=cXHvC_FGx24
3. scipy minimize function with parameters: https://stackoverflow.com/questions/43017792/minimize-function-with-parameters
---------
