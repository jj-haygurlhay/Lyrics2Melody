
from threading import Thread
from project_utils.BLEUscore import bleu_score
from project_utils.mmd import Compute_MMD

class Evaluator:
    def __init__(self, logger, max_n=2):
        self.logger = logger
        self.thread_list = []
        self.max_n = max_n
        self.init_results()
    
    def init_results(self):
        self.results = [None] * (3*self.max_n + 3)

    def evaluate_preds(self, epoch, train_loss, val_loss, pred_notes, pred_durations, pred_gaps, true_notes, true_durations, true_gaps):
        self.init_results()
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss

        id = 0
        # Compute BLEU scores
        for i in range(self.max_n):
            thread = Thread(target=self.get_bleu_scores, args=(pred_notes, true_notes, id, i, 'Notes'))
            thread.start()
            self.thread_list.append(thread)
            id += 1
        for i in range(self.max_n):
            thread = Thread(target=self.get_bleu_scores, args=(pred_durations, true_durations, id, i, 'Durations'))
            thread.start()
            self.thread_list.append(thread)
            id += 1
        for i in range(self.max_n):
            thread = Thread(target=self.get_bleu_scores, args=(pred_gaps, true_gaps, id, i, 'Gaps'))
            thread.start()
            self.thread_list.append(thread)
            id += 1

        # Compute MMD
        self.get_mmd(pred_notes, true_notes, id, 'Notes')
        note_mmd = self.results[id]
        id += 1
        thread = Thread(target=self.get_mmd, args=(pred_durations, true_durations, id, 'Durations'))
        thread.start()
        self.thread_list.append(thread)
        id += 1
        thread = Thread(target=self.get_mmd, args=(pred_gaps, true_gaps, id, 'Gaps'))
        thread.start()
        self.thread_list.append(thread)
        
        return note_mmd
    
    def get_bleu_scores(self, predicted, true, i, ngram, name):
        self.results[i] = bleu_score(predicted, true, max_n=ngram+1, weights=[1/(ngram+1)]*(ngram+1))
        self.logger.log_metric(f'BLEU/BLEU_{name}-{ngram+1}', self.results[i], self.epoch)

    def get_mmd(self, predicted, true, i, name):
        self.results[i] = Compute_MMD(predicted, true)
        self.logger.log_metric(f'MMD/MMD_{name}', self.results[i], self.epoch)

    def retrieve_results(self):
        if len(self.thread_list) > 0:
            self.wait_for_threads()

            bleu_scores_notes = self.results[:self.max_n]
            bleu_scores_durations = self.results[self.max_n:2*self.max_n]
            bleu_scores_gaps = self.results[2*self.max_n:3*self.max_n]
            mmd_notes = self.results[3*self.max_n]
            mmd_durations = self.results[3*self.max_n + 1]
            mmd_gaps = self.results[3*self.max_n + 2]

            val_results = {
                'epoch': self.epoch,
                'train_loss': self.train_loss,
                'val_loss': self.val_loss,
                'notes': {
                    'bleu': bleu_scores_notes + (5 - self.max_n) * [0],
                    'mmd': mmd_notes
                },
                'durations': {
                    'bleu': bleu_scores_durations + (5 - self.max_n) * [0],
                    'mmd': mmd_durations
                },
                'gaps': {
                    'bleu': bleu_scores_gaps + (5 - self.max_n) * [0],
                    'mmd': mmd_gaps
                },
            }
            self.logger.log_metric('Loss/Train', self.train_loss, self.epoch)
            self.logger.log_metric('Loss/Validation', self.val_loss, self.epoch)
            self.logger.log_results(val_results)
            print(val_results)

    def wait_for_threads(self):
        for thread in self.thread_list:
            thread.join()
        self.thread_list.clear()