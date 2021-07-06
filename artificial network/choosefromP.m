function [index] = choosefromP(P,l)
%pick a index from the given probability P which has length l
r=rand();
Psum=0;
for k=1:l
    Psum=Psum+P(k);
    if Psum>r
        index=uint16(k);
        break
    end
end
end

