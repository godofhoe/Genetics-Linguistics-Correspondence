function [word] = countword(w,N)
%count the frequency of each word
word=cell(N,2);
for i=1:N
    word{i,2}=0;
end
count=0;
for i=1:N
    appeared=0;
    for j=1:count
        if size(w{i})==size(word{j,1})
            if w{i}==word{j,1}
                appeared=1;
                word{j,2}=word{j,2}+1;
            end
        end
    end
    if appeared==0
        count=count+1;
        word{count,1}=w{i};
        word{count,2}=word{count,2}+1;
    end
end
word2=cell(count,2);
for i=1:count
    word2{i,1}=word{i,1};
    word2{i,2}=word{i,2};
end
word=word2;
end

