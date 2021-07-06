function [wordold] = oldword(samount,oamount,O,i,w)
%create a word with two old syllables
Gil=zeros(1,samount);
for l=1:samount
    for j=1:oamount
        if O(i,j)==1
            if w{j}(1)==l || w{j}(2)==l
                Gil(l)=Gil(l)+1;
            end
        end
    end
end
Pil=Gil/sum(Gil);
cursor=rand();
l=1;
s1=l;
while sum(Pil(1:l))<cursor
    l=l+1;
    s1=l;
end
s2=s1;
while s2==s1
    cursor=rand();
    l=1;
    s2=l;
    while sum(Pil(1:l))<cursor
        l=l+1;
        s2=l;
    end
end
if s2<s1
    wordold=[s2,s1];
else
    wordold=[s1,s2];
end
end

