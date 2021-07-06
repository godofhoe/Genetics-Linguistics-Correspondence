function [O] = randomnetwork3(N)
%create a random network with N nodes, each connection is random in (0,1)
O=zeros(N);
for i=2:N
    for j=1:i-1
        O(i,j)=rand();
        O(j,i)=O(i,j);
    end
end
end

