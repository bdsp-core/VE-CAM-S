function main_getFigures
    dataDir='./Data/';files=struct2cell(dir([dataDir,'*.mat']))';    
    trtDir='./eFigs/';
    if exist(trtDir,'dir')~=7
        mkdir(trtDir);
    end
    for k=1:size(files,1)
        file=files{k,1};tmp=load([dataDir,file]);
        data=tmp.data;start_time=datetime(tmp.start_time);Fs=tmp.Fs;caption=tmp.caption;score=tmp.score;
        f=fcn_plotEEG(data,start_time,Fs,score,caption);
        picName=[trtDir,strrep(file,'_data.mat','.png')];
        print(f,'-r600','-dpng',picName);     
        close;
    end  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function fig=fcn_plotEEG(data,start_time,Fs,score,caption)
        fig=figure('units','normalized','position',[0,0.0463,1.0000,0.8667]);
        ax=subplot('position',[.04,.050,.94,.938]);
        w=15;zScale=1/120;
        channel_withspace={'Fp1-F7','F7-T3','T3-T5','T5-O1','','Fp2-F8','F8-T4','T4-T6','T6-O2','','Fp1-F3','F3-C3','C3-P3','P3-O1','','Fp2-F4','F4-C4','C4-P4','P4-O2','','Fz-Cz','Cz-Pz','','EKG'};
        eeg=data(1:19,:);ekg=data(20,:);
        tto=1;tt1=w*Fs;tt=tto:tt1;gap=NaN(1,size(eeg,2));
        seg=fcn_Bipolar(eeg);
        seg_disp=[seg(1:4,:);gap;seg(5:8,:);gap;seg(9:12,:);gap;seg(13:16,:);gap;seg(17:18,:);gap;ekg];
        M=size(seg_disp,1);DCoff=repmat(flipud((1:M)'),1,size(seg_disp,2));
        seg_disp(seg_disp>300)=300;seg_disp(seg_disp<-300)=-300;
        ekg_=seg_disp(end,:);ekg_=(ekg_-mean(ekg_))/(eps+std(ekg_));
        set(fig,'CurrentAxes',ax);cla(ax)
        hold(ax,'on')
            text(ax,tt1,M+1,['\color{blue}VE-CAM-S score: ',num2str(score)],'fontsize',16,'FontWeight','bold','HorizontalAlignment','right','VerticalAlignment','middle')
            text(ax,0,-.9,['\color{blue}',caption],'fontsize',11,'HorizontalAlignment','left','VerticalAlignment','middle');
            for i=1:(w+1)
                ta=tto+Fs*(i-1);line(ax,[ta,ta],[0,M+1],'linestyle','--','color',[.5 .5 .5])
                text(ax,ta,-.5,datestr(start_time+seconds(i-1),'HH:MM:ss'),'HorizontalAlignment','center','VerticalAlignment','bottom','fontsize',9)
            end
            for iCh=1:length(channel_withspace)
                ta=DCoff(iCh);text(ax,-Fs/15,ta,channel_withspace(iCh),'fontsize',11,'FontWeight','bold','HorizontalAlignment','right','VerticalAlignment','middle')
            end
            plot(ax,tt,zScale*seg_disp(1:end-1,:)+DCoff(1:end-1,:),'k','linewidth',1);
            plot(ax,tt,.2*ekg_+DCoff(end,:),'color',[.5,.5,.5],'linewidth',1);axis off;
            dt=tt1-tto+1;a=round(dt*3.7/5);xa1=tto+[a,a+Fs-1];ya1=[5,5];xa2=tto+[a,a];ya2=ya1+[0,100*zScale];
            text(ax,xa1(1)-Fs*.05,mean(ya2),'100\muV','Color','b','FontSize',11,'HorizontalAlignment','right');
            text(ax,mean(xa1),4.7,'1 second','Color','b','FontSize',11,'HorizontalAlignment','center');
            line(ax,xa1,ya1,'LineWidth',1.5,'Color','b');
            line(ax,xa2,ya2,'LineWidth',1.5,'Color','b');
            set(ax,'ylim',[0,M+1],'xlim',[tto,tt1+1]);
        hold(ax,'off')
    end

    function dataBipolar=fcn_Bipolar(data)
        dataBipolar(9,:)=data(1,:)-data(2,:);
        dataBipolar(10,:)=data(2,:)-data(3,:);
        dataBipolar(11,:)=data(3,:)-data(4,:);
        dataBipolar(12,:)=data(4,:)-data(8,:);
        dataBipolar(13,:)=data(12,:)-data(13,:);
        dataBipolar(14,:)=data(13,:)-data(14,:);
        dataBipolar(15,:)=data(14,:)-data(15,:);
        dataBipolar(16,:)=data(15,:)-data(19,:);
        dataBipolar(1,:)=data(1,:)-data(5,:);
        dataBipolar(2,:)=data(5,:)-data(6,:);
        dataBipolar(3,:)=data(6,:)-data(7,:);
        dataBipolar(4,:)=data(7,:)-data(8,:);
        dataBipolar(5,:)=data(12,:)-data(16,:);
        dataBipolar(6,:)=data(16,:)-data(17,:);
        dataBipolar(7,:)=data(17,:)-data(18,:);
        dataBipolar(8,:)=data(18,:)-data(19,:);
        dataBipolar(17,:)=data(9,:)-data(10,:);
        dataBipolar(18,:)=data(10,:)-data(11,:);
    end
end