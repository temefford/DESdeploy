import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.setrecursionlimit(10000)


class G:
    ITERATIONS = 100
    DURATIONS = 6120

    
G.target_times = {
    1: 2,
    2: 3,
    3: 5
}


def update_target_times(target_times):
    G.target_times = {
        1: target_times[0],
        2: target_times[1],
        3: target_times[2]
    }

    
G.specialties = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5'
}


def update_globals(urg_times):
    G.process_times = {
                        1: urg_times[0],
                        2: urg_times[1],
                        3: urg_times[2]
                    }    
    

def create_arrival_times(sim_time, arr_rates):  #[time_between_urg 1 images, etc..]
    arrival_times_dict = {}
    arrival_times_tuples_list = []
    #Create arrival times for each urgency of images
    urg = 1
    for arr_time in arr_rates:
        arrival_times = []
        time = 0
        while time < sim_time:
            time += np.random.exponential(arr_time)
            arrival_times.append(time)
            arrival_times_tuples_list.append([urg, time])
        arrival_times_dict[urg] = arrival_times
        urg += 1
    arrival_times_tuples_list = sorted(arrival_times_tuples_list, key=lambda x: x[1])
    return arrival_times_dict, arrival_times_tuples_list


def create_medical_images(arrival_times_tuples_list):
    med_images = []
    for img_id, tup in enumerate(arrival_times_tuples_list):
        med_images.append(MedicalImage(img_id, tup[1], tup[0], random.sample(list(G.specialties.keys()), 1)[0]))
    print(f"{len(med_images)} medical images")
    return med_images


def create_constant_rads(num_rads):
    specialties_list = []
    for i in range(num_rads):
        specialties_list.append(random.sample(list(G.specialties.keys()), random.randrange(2,len(G.specialties))))
    G.const_specialities = specialties_list

    
def create_radiologists(num_rads, constant_rads):
    radiologists = []
    for i in range(num_rads):
        if constant_rads:
            specialties_temp = G.const_specialities[i] 
        else:
            specialties_temp = random.sample(list(G.specialties.keys()), random.randrange(2,len(G.specialties)))
        radiologists.append(Radiologist(i, specialties_temp))
    return radiologists


def create_initial_events(sim_duration, med_images, cutoff=False):
    events=[]
    for img in med_images:
        events.append([img.time_created, 'New Job', img])
    if cutoff:
        events.append([sim_duration*2, "Sim End"])
    events = sorted(events, key=lambda x: x[0])
    return events


def start_simulation(events, med_images, radiologists, constant_rads, cutoff=False):
    s = SystemState(events, med_images, G.radiologists, cutoff)
    s.run_simulation()


class MedicalImage(object):    
    def __init__(self, img_id, time_created, urgency, image_type):#, modality, speciality, urgency, image_label):
        self.img_id = img_id
        self.time_created = time_created
        self.urgency = urgency
        self.image_type = image_type
        self.target_time = G.target_times[urgency]
        self.time_remaining = G.target_times[urgency]
        self.est_process_time = G.process_times[urgency]
        self.in_queues = []   #keep track on which queues image is in [rad_id, position]
        self.time_seen = 0
        self.time_done = 0
        self.rad_seen = "None"
        
    def update_time_remaining(self, t):
        self.time_remaining = self.target_time - (t - self.time_created)
        
        
class Radiologist:
    def __init__(self, rad_id, specialties, working=True):
        self.queue = []
        self.queue_data = []#[med_image, image_id, image_urgency, time_left, est_time]
        self.rad_id = rad_id
        self.specialties = specialties
        self.is_working = working
        self.is_idle = 1
        self.images_served = []
        self.idle_times = []
        self.time_busy_start = 0
        self.time_idle_start = 0
        self.busy_times = []
        self.time = 0
        self.time_of_step = 0
        self.queue_length = []
        self.service_starts = []
        self.service_ends = []
        self.service_time = []  
        
    def get_stats(self):
        return self.idle_times, self.busy_times, self.queue_length, self.service_starts, self.service_ends, self.service_time 
        
    def show_queue(self):
        return self.queue
    
    def estimate_queue_time(self):
        t = 0
        for img in self.queue:
            t += img.est_process_time
        return t
    
    def add_job(self, med_image, time):
        #update idle time tracker
        if self.is_idle == 1:
            self.idle_times.append(time - self.time_idle_start)
            self.time_busy_start = time
        self.is_idle = 0
        #add img to queue and sort
        #self.queue.append(med_image)
        #if len(self.queue) > 1:
        #    self.sort_queue()
        if len(self.queue) <= 1:
            self.queue.append(med_image)
            self.queue_data.append([med_image, med_image.img_id, med_image.urgency, med_image.time_remaining, med_image.est_process_time, med_image.est_process_time]) #[image_id, image_urgency, time_left, est_time]
        else:
            if med_image.urgency == 1:
                ins_ind = 1
                for ind, img in enumerate(self.queue[1:]):
                    if img.urgency > 1:
                        ins_ind = ind + 1
                        break          
                self.queue.insert(ins_ind, med_image)
            elif med_image.urgency == 2:
                ins_ind = 1
                for ind, img in enumerate(self.queue[1:]):
                    if img.urgency > 2:
                        ins_ind = ind + 1
                        break          
                self.queue.insert(ins_ind, med_image)
            else:
                self.queue.append(med_image)

    def sort_queue(self):
        curr_img = self.queue[0]
        queue_tuple = [[img, img.urgency] for img in self.queue[1:]]
        queue_tuple = sorted(queue_tuple, key=lambda x: x[1])
        #print(queue_tuple)
        queue_tuple_list = [tup[0] for tup in queue_tuple]
        self.queue = queue_tuple_list
        self.queue.insert(0, curr_img)
       
    def finish_job(self, time):
        if len(self.queue) == 0:
            self.time_finished_last_job = time
            self.time_idle_start = time
            self.busy_times.append(time - self.time_busy_start)
            self.is_idle = 1
    
    def update_idle_lists(self, time):
        if self.is_idle == 1:
            self.idle_times.append(time - self.time_idle_start)
        elif self.is_idle == 0:
            self.busy_times.append(time - self.time_busy_start)
        
    def update_queue(self, time):
        for img in self.queue:
            img.update_time_remaining(time)
        #sort_queue()        
    #def sort_queue(self):
                   
        
        
class SystemState:
    def __init__(self, sim_duration, events, images, rads, cutoff=False, verbose=False):
        self.time = 0
        self.sim_duration = sim_duration
        self.continue_running = True
        self.events = events
        self.images = images
        self.rads = rads
        self.rads_working = rads
        self.rads_not_working = []
        self.events_history = []
        self.queue_lengths = []
        self.time_steps = []
        self.img_table = pd.DataFrame(columns=['img_id','urgency', 'create_time','seen_time', 'finished', 'img_wait_time', 'time_w_rad', 'total_time'])
        self.rad_table = pd.DataFrame()
        self.unfin_img_table = pd.DataFrame(columns=['img_id','urgency', 'create_time','seen_time', 'finished', 'img_wait_time', 'time_w_rad', 'total_time'])
        self.verbose = verbose
        
    def create_event(self, time, event_type, obj):
        self.events.append([time, event_type, obj])
        self.events = sorted(self.events, key=lambda x: x[0])

    def update_img_table(self, med_img):
        column_names = ['img_id','urgency', 'create_time','seen_time', 'finished','img_wait_time', 'time_w_rad', 'total_time']
        values = [med_img.img_id, med_img.urgency, med_img.time_created, med_img.time_seen, self.time, med_img.time_seen - med_img.time_created, self.time - med_img.time_seen, self.time - med_img.time_created] #[[med_img.time_created], [med_img.time_seen], [self.time], [self.time - med_img.time_seen], [self.time - med_img.time_created]]
        temp_df = pd.DataFrame(values).T
        temp_df.columns = column_names
        self.img_table = self.img_table.append(temp_df, ignore_index = True)
        
    def unfinished_jobs(self):
        unfin_med_images = []
        for rad in self.rads:
            unfin_med_images += rad.queue
            unfin_med_images = list(set(unfin_med_images))
        print(f"There are {len(unfin_med_images)} that were not completed in time")
        column_names = ['img_id','urgency', 'create_time','seen_time', 'finished','img_wait_time', 'time_w_rad', 'total_time']
        for med_img in unfin_med_images:
            values = [med_img.img_id, med_img.urgency, med_img.time_created, med_img.time_seen, self.time, med_img.time_seen - med_img.time_created, self.time - med_img.time_seen, self.time - med_img.time_created] #[[med_img.time_created], [med_img.time_seen], [self.time], [self.time - med_img.time_seen], [self.time - med_img.time_created]]
            temp_df = pd.DataFrame(values).T
            temp_df.columns = column_names
            self.unfin_img_table = self.unfin_img_table.append(temp_df, ignore_index = True)
        
    def process_event(self):
        event = self.events[0]
        self.events_history.append(event)
        self.time = event[0]       
        event_type = event[1]
        del self.events[0]
        temp_list = []
        for r in self.rads:
            temp_list.append(len(r.queue))
        self.queue_lengths.append(temp_list)
        self.time_steps.append(self.time)        
            
        if event_type == "New Job":
            self.distribute_job(event[2])
        elif event_type == "Job Done":
            rad = event[2]
            self.complete_job(rad)
        elif event_type == "Sim End":
            self.continue_running = False 
        if self.verbose==True:
            print("Event processed")
        if (len(self.events) == 0) or (self.events[0][1]=="Sim End"):
            self.continue_running = False 
        if self.continue_running:
            self.process_event()
        else:
            for rad in self.rads:
                rad.update_idle_lists(self.time)
            self.unfinished_jobs()
            print(f"Simulation complete at {self.time} minutes")
                
    def distribute_job(self, med_image):
        urgency = med_image.urgency
        image_type = med_image.image_type
        # Function to route medical images based on some algorithm
        chosen_rads = self.choose_rads(image_type)       
        for rad in chosen_rads:
            rad.add_job(med_image, self.time)
            med_image.in_queues.append(rad)    #keep track of which rads have image in queue
            if len(rad.queue)==1:
                self.start_job(rad)
                break         
        self.update_queues() 
        
    def choose_rads(self, image_type):
        capable_rads = []
        for rad in self.rads_working:      #finds radiologists capable of working on image
            if image_type in rad.specialties:
                capable_rads.append(rad)
        chosen_rads = self.n_quickest_queues(capable_rads, 3)
        return capable_rads
    
    def n_shortest_queues(self, rads_list, n):
        rads_tuples = []
        for rad in rads_list:
            rads_tuples.append([rad, len(rad.queue)])
        rads_tuples.sort(key = lambda x: x[1])
        return [rad[0] for rad in rads_tuples[:n]]
    
    def n_quickest_queues(self, rads_list, n):
        rads_tuples = []
        for rad in rads_list:
            rads_tuples.append([rad, rad.estimate_queue_time()])
        rads_tuples.sort(key = lambda x: x[1])
        return [rad[0] for rad in rads_tuples[:n]]
             
    def update_queues(self):
        for rad in self.rads_working:
            rad.update_queue(self.time)
                
    def start_job(self, rad):
        med_image = rad.queue[0]
        image_type = med_image.image_type
        urgency = med_image.urgency
        rad.service_starts = self.time
        med_image.time_seen = self.time
        med_image.rad_seen = rad.rad_id
        self.events_history.append([self.time, "Job Started", med_image])
        process_time = np.random.exponential(G.target_times[urgency])
        self.create_event(self.time+process_time, "Job Done", rad)
        if self.verbose==True:
            print(f"Image {med_image.img_id} is seen by radiologist {rad.rad_id} at {self.time}")
        for r in med_image.in_queues:
            if r != rad:
                r.queue.remove(med_image)           
        
    def complete_job(self, rad):
        med_image = rad.queue[0]
        self.update_img_table(med_image)
        rad.images_served.append(med_image.img_id)
        rad.service_ends.append(self.time)
        med_image.time_done = self.time
        if self.verbose==True:
            print(f"Image {med_image.img_id} is done by radiologist {rad.rad_id} at {self.time}")
        del rad.queue[0]
        rad.finish_job(self.time)
        if len(rad.queue) > 0:
            self.start_job(rad)

    def run_simulation(self):
        self.process_event()