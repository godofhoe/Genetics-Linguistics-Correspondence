function [syllable] = syllablefrequency(N,w)
%store the syllable, frequency in appearance order
count=0;
syllable=zeros(N,2);
for i=1:N
    l=length(w{i});
    for j=1:l
        found=0;
        for k=1:count
            if syllable(k,1)==w{i}(j)
                found=1;
                syllable(k,2)=syllable(k,2)+1;
                break
            end
        end
        if found==0
            count=count+1;
            syllable(count,1)=w{i}(j);
            syllable(count,2)=syllable(count,2)+1;
        end
    end
end
syllable=syllable(1:count,:);
end

