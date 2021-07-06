function [wordrank] = rankword(word)
%rank the words
[m,~]=size(word);
sample=zeros(m,1);
for i=1:m
    sample(i)=word{i,2}-i/(1+m);
end
[~,index]=sort(sample,'descend');
[~,rank]=sort(index);
wordrank=rank;
end

