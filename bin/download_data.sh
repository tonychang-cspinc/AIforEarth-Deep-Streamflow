#/bin/bash
DATADIR=/datadrive
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/Grant_Poole_documents_20200304.zip -P $DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/MA%20DER.zip -P $DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/MA%20DER%20part2.zip -P  $DATADIR
#noflow files
NOFLOW_DATADIR=/datadrive/noflow
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20180613_20180808_LakeSamoset.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20180613_20190519_TaftPond.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20180724_20190409_OldGristMill.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20180724_20191204_OtisReservoir.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20180809_20191023_LakeSamoset.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20190807_20191008_BrownsBrook.zip -P $NOFLOW_DATADIR
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/20190909_20191204_ConeBrook.zip -P $NOFLOW_DATADI
wget http://walkerenvres.com.s3.us-east-1.amazonaws.com/share/sheds/fpe/FDL%20Creek%20-%20Early%20July%202019.zip -P $NOFLOW_DATADIR
