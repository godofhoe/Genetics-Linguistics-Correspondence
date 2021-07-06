function [syllablerank] = ranksyllable(syllable)
%calculate the rank of each syllable
[m,~]=size(syllable);
sample=zeros(m,1);
for i=1:m
    sample(i)=syllable(i,2)-i/(m+1);
end
[~,index]=sort(sample,'descend');
[~,rank]=sort(index);
syllablerank=syllable;
syllablerank(:,2)=rank;
end

