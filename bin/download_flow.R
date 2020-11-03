library(dplyr)
library(waterData)
getEnvData <- function( startDate = '2016-12-18', gage = '01169900'){
	################################################
	# get environmental data
	# gage from // south river, conway
		 
	flowData <- importDVs(gage, code = "00060", stat = "00003", sdate = startDate)
	flowData <- dplyr:::rename(flowData, flow = val )

	flowData$date <- as.character(flowData$dates)
return(flowData)
}
