'''
Forecasting Time series data
'''
import time, datetime
import logging
import yaml
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
#TODO develop function to handle outliers for use with fixed timestep models and for possible benchmarking


class TSF_Service(object):
    '''
    Class for real world deployment of TS_Models
    '''

    def __init__(self,admin_yaml,model_yaml,data_yaml):
        # Logging
        self.init_logger()

        # Admins
        self.admin_yaml = admin_yaml
        self.init_admins(self.admin_yaml)

        # Models
        self.model_yaml = model_yaml
        self.init_models(self.model_yaml)

        # Data
        self.data_yaml = data_yaml
        self.init_data(self.data_yaml)

        self.logger.info('Successful Initialization of TSFS')

    def init_admins(self,admin_yaml):
        try:
            self.admin_emails = yaml.load(open(admin_yaml))
            self.logger.info('Admin Accounts Initialized')
        except Exception as e:
            self.logger.info('Failed to load Admin Yaml: %s, Error: %s',admin_yaml,e)
        pass

    def init_models(self,model_yamls):
        self.models = {}
        pass

    def init_data(self,data_yaml):
        pass

    def init_logger(self):
        # create logger
        self.logger = logging.getLogger('TSF Service Logger')
        self.logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create file handler and set level to debug
        hdlr = logging.FileHandler('persistent_tsfs.log',mode='a')
        hdlr.setFormatter(formatter)
        hdlr.setLevel(logging.DEBUG)

        # create file handler and set level to debug
        self.temp_log = 'temp_tsfs.txt' # .txt extension makes it previewable in google docs
        temp = logging.FileHandler(self.temp_log,mode='w')
        temp.setFormatter(formatter)
        temp.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(hdlr)
        self.logger.addHandler(temp)
        self.logger.info('Logger Initialized')

    def alert_admins(self,alert):
        self.logger.info('START Alerting Admins: %s'%self.admin_emails)
        for admin in self.admin_emails:
            try:
                self.email(self.admin_emails[admin],alert,name=admin)
            except Exception as e:
                self.logger.error(e)
        self.logger.info('END Alerting Admins')

    def email(self,adr,text,name='MSI employee'):
        ''' send and email '''
        try:
            # TODO ----- Send email here -----
            fromaddr = "timeseriesforecaster@gmail.com"
            toaddr = adr
            msg = MIMEMultipart()
            msg['From'] = fromaddr
            msg['To'] = toaddr
            msg['Subject'] = "MSI TimeSeries Forecasting Service"

            body = "Hello %s,\n\n" \
                   "This is an automated email from TimeSeries Forecaster Service (TSFS).\n" \
                   "Body: %s\n" \
                   "A complete log of TSFS\'s activities can be found on the EC2 instance that it runs on.\n" \
                   "\nHappy Forecasting,\n\t-TSFS"%(name,text)

            msg.attach(MIMEText(body, 'plain'))

            filename = self.temp_log
            attachment = open(self.temp_log, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            msg.attach(part)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(fromaddr, "MSI4life!")
            text = msg.as_string()
            server.sendmail(fromaddr, toaddr, text)
            server.quit()
            self.logger.info('Sent email to %s', adr)
        except Exception as e:
            self.logger.error('Failed to send email to %s. Error: %s',adr,e)

    def test_alert(self):
        self.logger.info('START: testing alert system')
        self.admin_emails['NotDeliverable'] = 'invalid email address'
        self.alert_admins('This is a Test of the Admin Alert Function')
        self.admin_emails.pop('NotDeliverable',None)
        self.logger.info('END: testing alert system')
        return True

    def update(self):
        # Get latest data from source database

        # Check for anomalous data

        # Refit any models that need it

        # Get new forecasts

        # Dump new forecasts to ????

        self.logger.info('Completed Update')
        self.alert_admins('Hooray, TSFS completed an Update!')


def test_ts_model_class(f):
    '''
    :param f:
    :return:
    '''

    return True or False






class TS_Bencharmk(object):
    ''' Class for comparing TS_Models '''

    def __init__(self,datapath):
        self.init_timestamp = datetime.datetime.now()
        self.datapath = datapath
        # Add code here to load and pre-process data
        #(Fixing outliers, centering, whitening, converting variable types, composing features, etc)
        pass

    def __str__(self):
        return "TS Benchmark Instance\n" \
               "\tDateTime initialized: %s" \
               "\tOriginal data path: %s"\
               %(self.init_timestamp,
                 self.datapath)

    def gen_train_test(self,seed=1234):
        pass

    def compare_models(self,dict_of_models,out_dir):
        '''
        Compares models even if they aren't trained. Useful if data changes and we need to compare multiple models
        :param dict_of_models:
        :param out_dir:
        :return:
        '''
        # Test that each model follows TS_Model prototype
        #todo

        # Try training each model on the train data set
        #todo

        # Perform Forecast on each model
        #todo

        # Pass forecast files to self.compare_forecasts
        #todo

        pass

    def compare_forecasts(self,yaml_file,out_dir):
        '''
        Compare forecast files specified by yaml
        :param yaml_file: Describes models and their forecast predictions
        :param out_dir: Dump path for analysis files
        :return: Nothing
        '''
        #todo

        pass

    def score_forecast(self,file):
        ''' Given a forecast file, compute accuracy measures '''
        # test format
        # Score file
        pass

    def summary(self,file=None):
        ''' Given that a comparison has already been complete, print or write to file a summary report'''
        pass

    def summary_viz(self,dir=None):
        ''' Given that a comparison has already been complete, show or save summary figures to a directory '''
        pass

    def test_format(self,file):
        '''
        Test if forecast file is in correct format and matches object instance
        :param file:
        :return:
        '''
        return True



if __name__ == '__main__':
    # Example of TS_Benchmark Class
    # Initializing it with data for the first time

    # Get training and testing data (No one should get test data until later in hackathon)

    # Saving object instance to disk so it can be used again

    # Loading object instance so it can be used at a later time
    print('yo')
