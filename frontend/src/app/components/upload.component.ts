import { Component, inject, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService, ProgressUpdate, AnalysisResults } from '../services/api.service';
import { ProgressComponent } from './progress.component';
import { ResultsComponent } from './results.component';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, ProgressComponent, ResultsComponent],
  template: `
    <div class="space-y-8">
      <!-- Upload Section -->
      <div *ngIf="!isProcessing && !results" class="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 p-8 hover:shadow-purple-500/20 transition-all duration-300">
        <div class="mb-6">
          <h2 class="text-3xl font-bold text-white mb-2">Upload Check Image</h2>
          <p class="text-gray-300">Drag and drop or click to upload a check image for advanced fraud detection analysis</p>
        </div>

        <!-- Drag and Drop Area -->
        <div
          [class]="
            'relative border-3 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 group overflow-hidden ' +
            (isDragging ? 'border-purple-400 bg-purple-500/10' : 'border-white/30 hover:border-purple-400 hover:bg-white/5')
          "
          (dragover)="onDragOver($event)"
          (dragleave)="onDragLeave($event)"
          (drop)="onDrop($event)"
          (click)="fileInput.click()"
        >
          <div class="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-purple-500/5 to-pink-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <div class="relative flex flex-col items-center">
            <div class="mb-6 relative">
              <div class="absolute inset-0 bg-purple-500/20 rounded-full blur-xl group-hover:bg-purple-500/30 transition-all"></div>
              <svg class="w-20 h-20 text-purple-400 relative animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
            </div>
            <p class="text-xl text-white mb-2">
              <span class="font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Click to upload</span>
              <span class="text-gray-300"> or drag and drop</span>
            </p>
            <p class="text-sm text-gray-400 mb-4">JPEG, PNG, or PDF • Maximum 10MB</p>
            <div class="flex items-center space-x-4 text-xs text-gray-500">
              <span class="flex items-center"
                ><svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" /></svg
                >Secure Upload</span
              >
              <span class="flex items-center"
                ><svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6z" /></svg>AI
                Analysis</span
              >
            </div>
          </div>
          <input #fileInput type="file" class="hidden" accept=".jpg,.jpeg,.png,.pdf" (change)="onFileSelected($event)" />
        </div>

        <!-- Selected File Info -->
        <div *ngIf="selectedFile" class="mt-6 p-5 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-purple-500/30 backdrop-blur-sm">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  ></path>
                </svg>
              </div>
              <div>
                <p class="font-semibold text-white">{{ selectedFile.name }}</p>
                <p class="text-sm text-gray-400">{{ formatFileSize(selectedFile.size) }}</p>
              </div>
            </div>
            <button (click)="clearFile(); $event.stopPropagation()" class="text-red-400 hover:text-red-300 hover:bg-red-500/10 p-2 rounded-lg transition-colors">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>

        <!-- Upload Button -->
        <button
          *ngIf="selectedFile"
          (click)="uploadFile()"
          class="mt-6 w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-8 rounded-xl font-bold text-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-purple-500/50 hover:scale-[1.02] active:scale-[0.98]"
        >
          <span class="flex items-center justify-center">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
            Start AI Analysis
          </span>
        </button>
      </div>

      <!-- Progress Section -->
      <app-progress *ngIf="isProcessing" [progress]="progress" [currentStage]="currentStage" [statusMessage]="statusMessage"></app-progress>

      <!-- Results Section -->
      <app-results *ngIf="results" [results]="results" [jobId]="jobId" (newAnalysis)="resetForNewAnalysis()"></app-results>
    </div>
  `,
  styles: [],
})
export class UploadComponent {
  private apiService = inject(ApiService);
  private ngZone = inject(NgZone);

  selectedFile: File | null = null;
  isDragging = false;
  isProcessing = false;

  // Progress tracking
  progress = 0;
  currentStage = 0;
  statusMessage = '';

  // Results
  jobId = '';
  results: AnalysisResults | null = null;

  private eventSource: EventSource | null = null;

  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFile(input.files[0]);
    }
  }

  handleFile(file: File) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
      alert('Please upload a JPEG, PNG, or PDF file');
      return;
    }

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      alert('File size must be less than 10MB');
      return;
    }

    this.selectedFile = file;
  }

  clearFile() {
    this.selectedFile = null;
  }

  uploadFile() {
    if (!this.selectedFile) return;

    this.isProcessing = true;
    this.progress = 0;
    this.currentStage = 0;
    this.statusMessage = 'Uploading check...';

    this.apiService.uploadCheck(this.selectedFile).subscribe({
      next: (response) => {
        this.jobId = response.jobId;
        this.listenToProgress(response.jobId);
      },
      error: (error) => {
        console.error('Upload error:', error);
        alert('Failed to upload check. Please try again.');
        this.isProcessing = false;
      },
    });
  }

  listenToProgress(jobId: string) {
    this.eventSource = this.apiService.listenToProgress(
      jobId,
      (update: ProgressUpdate) => {
        this.ngZone.run(() => {
          this.progress = update.currentPercentage;
          this.currentStage = update.currentStage;
          this.statusMessage = update.message;

          // Check if completed
          if (update.status === 'COMPLETED' && update.currentPercentage === 100) {
            setTimeout(() => {
              this.fetchResults(jobId);
            }, 500);
          }
        });
      },
      () => {
        // Analysis complete, fetch results
        this.ngZone.run(() => {
          this.fetchResults(jobId);
        });
      },
      (error) => {
        console.error('Progress stream error:', error);
        this.ngZone.run(() => {
          this.isProcessing = false;
          alert('Connection lost. Please refresh and try again.');
        });
      }
    );
  }

  fetchResults(jobId: string) {
    this.apiService.getResults(jobId).subscribe({
      next: (results) => {
        this.ngZone.run(() => {
          // Check if analysis is actually complete
          if (results.status === 'COMPLETED') {
            this.results = results;
            this.isProcessing = false;

            // Close event source
            if (this.eventSource) {
              this.eventSource.close();
              this.eventSource = null;
            }
          } else {
            // Still processing, wait a bit and try again
            setTimeout(() => this.fetchResults(jobId), 1000);
          }
        });
      },
      error: (error) => {
        console.error('Error fetching results:', error);
        this.ngZone.run(() => {
          this.isProcessing = false;
          alert('Failed to fetch results. Please try again.');
        });
      },
    });
  }

  resetForNewAnalysis() {
    this.ngZone.run(() => {
      this.selectedFile = null;
      this.isProcessing = false;
      this.progress = 0;
      this.currentStage = 0;
      this.statusMessage = '';
      this.jobId = '';
      this.results = null;

      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
    });
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  }

  ngOnDestroy() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  }
}
