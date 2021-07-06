    close all
N=10000;      %amount of nodes as objects
z=0.1*N;      %portion of new syllable
lambda=0.495; %portion of listener in total effort
P=0.0005;      %probability of refreshing
[O1]=randomnetwork3(N);     %network of first syllable
[O2]=randomnetwork3(N);     %network of second syllable
[O3]=randomnetwork3(N);     %network of third syllable
[O4]=randomnetwork3(N);     %network of fourth syllable
[O5]=randomnetwork3(N);     %network of fifth syllable

% O1=zeros(N);
% O1(:,:)=0.5;
% for i=1:N
%     O1(i,i)=0;
% end
% O2=O1;
% O3=O1;
% O4=O1;
% O5=O1;

O6=O1+O2+O3+O4+O5;

w=cell(N,1);                %word's index and syllables
w{1}=[1,1,2];               %first word
w{2}=[2,2,1];               %second word
Ns=uint64(2);               %amount of existing syllable
Nw=2;                       %amount of existing word
Pold=zeros(N,1);            %probability for each old word to be choose
Psnew=zeros(N+1,1);         %probability for syllable in new word

Pw=zeros(N,1);              %probability to use word w
Pw(1)=0.5;
Pw(2)=0.5;
Hs=-2*1/2*log(1/2)/log(N);  %effort of speaker
Hhw=zeros(N,1);             %effort of hearer hearing word w
Hh=0;                       %effort of hearer

for i=3:N
    %choose the old word
    sumO=sum(O6(i,1:i-1));  %nomalization factor for P
    for j=1:i-1
        Pold(j)=O6(i,j)/sumO;
    end
    woldobjectindex=choosefromP(Pold,i-1);  %index of the choosen old word
    oldword=w{woldobjectindex};
    l=oldword(1);
    
    %calculate effort for choosing old word
    Pwl=Pw(l);
    Pwlnew=(i-1)/i*Pwl+1/i;
    Hsold=(i-1)/i*(Hs+(Pwl-1)*log((i-1)/i)/log(N)+Pwl*log(Pwl)/log(N))-Pwlnew*log(Pwlnew)/log(N);
    Hhwl=Hhw(l);
    Hhwlnew=log((i-1)^Hhwl+1)/log(i);
    Hhold=(i-1)/i*(Hh-Pwl*Hhwl)+Pwlnew*Hhwlnew;
    Eold=lambda*Hhold+(1-lambda)*Hsold;
    
    %calculate effort for choosing new word
    Hsnew=1/i*((i-1)*Hs-(i-1)*log((i-1)/i)/log(N)+log(i)/log(N));
    Hhnew=(i-1)*Hh/i/log(i)*log(i-1);
    Enew=lambda*Hhnew+(1-lambda)*Hsnew;
    
    %decide to use old word or new word
    Pw=(i-1)/i*Pw;
    if  Eold<Enew
        Pw(l)=Pw(l)+1/i;
        Hs=Hsold;
        Hhw(l)=log((i-1)^Hhwl+1)/log(i);
        Hh=Hhold;
        w{i}=oldword;
    else
        %choose the new word
        exist=true;              %make sure the new word is new
        while exist
            exist=false;
            
            %decide the length of new word
            index=rand();
            if index>0.15
                L=2;
                if index>0.55
                    L=3;
                    if index>0.8
                        L=4;
                        if index>0.95
                            L=5;
                        end
                    end
                end
            else
                L=1;
            end
            
            newword=zeros(1,L+1,'uint16');     %new word
            
            %choose first syllable of the new word
            sumO=sum(O1(i,1:i-1))+z;
            for j=1:i-1
                Psnew(j)=O1(i,j)/sumO;
            end
            Psnew(i)=z/sumO;
            snew=choosefromP(Psnew,i);   %index for syllable of new word
            if snew==i
                newword(2)=Ns+1;
            else
                newword(2)=w{snew}(2);
            end
            
            if L>1
                %choose the second syllable of the new word
                for j=1:i-1
                    if length(w{j})>2
                        Psnew(j)=O2(i,j);
                    else
                        Psnew(j)=0;
                    end
                end
                Psnew(i)=z;
                Psnew=Psnew/sum(Psnew);
                snew=choosefromP(Psnew,i);
                if snew==i
                    newword(3)=Ns+1;
                else
                    newword(3)=w{snew}(3);
                end
                if L>2
                    %choose the third syllable of the new word
                    for j=1:i-1
                        if length(w{j})>3
                            Psnew(j)=O3(i,j);
                        else
                            Psnew(j)=0;
                        end
                    end
                    Psnew(i)=z;
                    Psnew=Psnew/sum(Psnew);
                    snew=choosefromP(Psnew,i);
                    if snew==i
                        newword(4)=Ns+1;
                    else
                        newword(4)=w{snew}(4);
                    end
                    if L>3
                        %choose the fourth syllable of the new word
                        for j=1:i-1
                            if length(w{j})>4
                                Psnew(j)=O4(i,j);
                            else
                                Psnew(j)=0;
                            end
                        end
                        Psnew(i)=z;
                        Psnew=Psnew/sum(Psnew);
                        snew=choosefromP(Psnew,i);
                        if snew==i
                            newword(5)=Ns+1;
                        else
                            newword(5)=w{snew}(5);
                        end
                        if L>4
                            %choose the fifth syllable of the new word
                            for j=1:i-1
                                if length(w{j})>5
                                    Psnew(j)=O5(i,j);
                                else
                                    Psnew(j)=0;
                                end
                            end
                            Psnew(i)=z;
                            Psnew=Psnew/sum(Psnew);
                            snew=choosefromP(Psnew,i);
                            if snew==i
                                newword(6)=Ns+1;
                            else
                                newword(6)=w{snew}(6);
                            end
                        end
                    end
                end
            end
            
            newword(1)=Nw+1;
            
            for j=1:i-1
                if length(w{j})==L+1
                    if newword(2:L)==w{j}(2:L)
                        exist=true;
                    end
                end
            end
        end
        
        Pw(Nw+1)=1/i;
        Hs=Hsnew;
        Hhw=Hhw/log(i)*log(i-1);
        Hh=Hhnew;
        if L==1
            if newword(2)>Ns
                Ns=Ns+1;
            end
        else
            if L==2
                if newword(2)>Ns||newword(3)>Ns
                    Ns=Ns+1;
                end
            else
                if L==3
                    if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns
                        Ns=Ns+1;
                    end
                else
                    if L==4
                        if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns||newword(5)>Ns
                            Ns=Ns+1;
                        end
                    else
                        if L==5
                            if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns||newword(5)>Ns||newword(6)>Ns
                                Ns=Ns+1;
                            end
                        end
                    end
                end
            end
        end
        Nw=Nw+1;
        w{i}=newword;
    end
    
    %refresh with principle of least effort
    for l=1:i
        if rand()<P
            p=w{l}(1);                  %word assigned to ol
            woldobjectindex=choosefromP(Pold,i);  %index of the choosen object
            oldword=w{woldobjectindex};
            q=oldword(1);                           %choosen word
            
            %calculate effort if old word was chose
            Pwp=Pw(p);
            Pwpnew=Pwp-1/i;
            if Pwpnew<0
                Pwpnew=0;
            end
            Pwq=Pw(q);
            Pwqnew=Pwq+1/i;
            Hsold=Hs+Pwp*log(Pwp)/log(N)-Pwqnew*log(Pwqnew)/log(N);
            if Pwq~=0
                Hsold=Hsold+Pwq*log(Pwq)/log(N);
            end
            if Pwpnew~=0
                Hsold=Hsold-Pwpnew*log(Pwpnew)/log(N);
            end
            Hhwp=Hhw(p);
            if Hhwp~=0
                Hhwpnew=log(N^Hhwp-1)/log(N);
            else
                Hhwpnew=0;
            end
            Hhwq=Hhw(q);
            Hhwqnew=log(N^Hhwq+1)/log(N);
            Hhold=Hh-Pwp*Hhwp-Pwq*Hhwq+Pwpnew*Hhwpnew+Pwqnew*Hhwqnew;
            Eold=lambda*Hhold+(1-lambda)*Hsold;
            
            %calculate effort if new word was chose
            Hsnew=Hs+Pwp*log(Pwp)/log(N)+1/i*log(i)/log(N);
            if Pwpnew~=0
                Hsnew=Hsnew-Pwpnew*log(Pwpnew)/log(N);
            end
            Hhnew=Hh-Pwp*Hhwp+Pwpnew*Hhwpnew;
            Enew=lambda*Hhnew+(1-lambda)*Hsnew;
            E=lambda*Hh+(1-lambda)*Hs;
            %make decision
            if Eold<Enew && Eold<E
                Pw(p)=Pwpnew;
                Hhw(p)=Hhwpnew;
                Pw(q)=Pwqnew;
                Hs=Hsold;
                Hhw(q)=Hhwqnew;
                Hh=Hhold;
                w{l}=oldword;
            else
                if Enew<Eold && Enew<E
                    %choose the new word
                    exist=true;              %make sure the new word is new
                    while exist
                        exist=false;
                        
                        %decide the length of new word
                        index=rand();
                        if index>0.15
                            L=2;
                            if index>0.55
                                L=3;
                                if index>0.8
                                    L=4;
                                    if index>0.95
                                        L=5;
                                    end
                                end
                            end
                        else
                            L=1;
                        end
                        
                        newword=zeros(1,L+1,'uint16');     %new word
                        
                        %choose first syllable of the new word
                        sumO=sum(O1(i,1:i-1))+z;
                        for j=1:i-1
                            Psnew(j)=O1(i,j)/sumO;
                        end
                        Psnew(i)=z/sumO;
                        snew=choosefromP(Psnew,i);   %index for syllable of new word
                        if snew==i
                            newword(2)=Ns+1;
                        else
                            newword(2)=w{snew}(2);
                        end
                        
                        if L>1
                            %choose the second syllable of the new word
                            for j=1:i-1
                                if length(w{j})>2
                                    Psnew(j)=O2(i,j);
                                else
                                    Psnew(j)=0;
                                end
                            end
                            Psnew(i)=z;
                            Psnew=Psnew/sum(Psnew);
                            snew=choosefromP(Psnew,i);
                            if snew==i
                                newword(3)=Ns+1;
                            else
                                newword(3)=w{snew}(3);
                            end
                            if L>2
                                %choose the third syllable of the new word
                                for j=1:i-1
                                    if length(w{j})>3
                                        Psnew(j)=O3(i,j);
                                    else
                                        Psnew(j)=0;
                                    end
                                end
                                Psnew(i)=z;
                                Psnew=Psnew/sum(Psnew);
                                snew=choosefromP(Psnew,i);
                                if snew==i
                                    newword(4)=Ns+1;
                                else
                                    newword(4)=w{snew}(4);
                                end
                                if L>3
                                    %choose the fourth syllable of the new word
                                    for j=1:i-1
                                        if length(w{j})>4
                                            Psnew(j)=O4(i,j);
                                        else
                                            Psnew(j)=0;
                                        end
                                    end
                                    Psnew(i)=z;
                                    Psnew=Psnew/sum(Psnew);
                                    snew=choosefromP(Psnew,i);
                                    if snew==i
                                        newword(5)=Ns+1;
                                    else
                                        newword(5)=w{snew}(5);
                                    end
                                    if L>4
                                        %choose the fifth syllable of the new word
                                        for j=1:i-1
                                            if length(w{j})>5
                                                Psnew(j)=O5(i,j);
                                            else
                                                Psnew(j)=0;
                                            end
                                        end
                                        Psnew(i)=z;
                                        Psnew=Psnew/sum(Psnew);
                                        snew=choosefromP(Psnew,i);
                                        if snew==i
                                            newword(6)=Ns+1;
                                        else
                                            newword(6)=w{snew}(6);
                                        end
                                    end
                                end
                            end
                        end
                        
                        newword(1)=Nw+1;
                        
                        for j=1:i-1
                            if length(w{j})==L+1
                                if newword(2:L)==w{j}(2:L)
                                    exist=true;
                                end
                            end
                        end
                    end
                    
                    Pw(Nw+1)=1/i;
                    Hs=Hsnew;
                    Hhw=Hhw/log(i)*log(i-1);
                    Hh=Hhnew;
                    if L==1
                        if newword(2)>Ns
                            Ns=Ns+1;
                        end
                    else
                        if L==2
                            if newword(2)>Ns||newword(3)>Ns
                                Ns=Ns+1;
                            end
                        else
                            if L==3
                                if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns
                                    Ns=Ns+1;
                                end
                            else
                                if L==4
                                    if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns||newword(5)>Ns
                                        Ns=Ns+1;
                                    end
                                else
                                    if L==5
                                        if newword(2)>Ns||newword(3)>Ns||newword(4)>Ns||newword(5)>Ns||newword(6)>Ns
                                            Ns=Ns+1;
                                        end
                                    end
                                end
                            end
                        end
                    end
                    Nw=Nw+1;
                    
                    w{l}=newword;
                end
            end
            
            %check if the replaced word still exist
            exist=false;
            for j=1:i
                if w{j}(1)==p
                    exist=true;
                end
            end
            if exist==false
                for j=1:i
                    if w{j}(1)>p
                        w{j}(1)=w{j}(1)-1;
                    end
                end
                for j=p:Nw
                    Pw(j)=Pw(j+1);
                    Hhw(j)=Hhw(j+1);
                end
                Nw=Nw-1;
            end
        end
    end
end

%swap order randomly
w2=w;
order=rand(N,1);
[~,index]=sort(order);
for i=1:N
    w{i}=w2{index(i)};
end

%check zipf's law
[word]=countword(w,N);
[m,~]=size(word);
f=zeros(m,1);
for i=1:m
    f(i)=word{i,2};
end
[fr]=sort(f,'descend');
x=1:1:m;
figure;
loglog(x,fr,'ok');
zips=gcf;               %handle of zip's plot
set(zips,'position',[910,430,450,250]);
xlabel('rank');
ylabel('frequency');
axis([-inf,inf,-inf,inf]);

%check RRD
[syllable]=syllablefrequency(N,w);
[syllablerank]=ranksyllable(syllable);
[wordrank]=rankword(word);

[m,~]=size(word);
[l,~]=size(syllable);
xy=zeros(N*N,1);
count=0;
for i=1:m
    [~,n]=size(word{i,1});
    for j=1:n
        found=0;
        for k=1:l
            if syllablerank(k,1)==word{i,1}(j)
                count=count+1;
                xy(count,2)=syllablerank(k,2);
                xy(count,1)=wordrank(i);
            end
        end
    end
end
xy=xy(1:count,:);
figure;
plot(xy(:,1),xy(:,2),'.k');
rrd=gcf;
set(rrd,'position',[910,40,450,320]);
xlabel('word');
ylabel('syllable');
axis([-inf,inf,-inf,inf]);