import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-progress',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 p-8 hover:shadow-purple-500/20 transition-all duration-300">
      <div class="mb-8">
        <h2 class="text-3xl font-bold text-white mb-2 flex items-center">
          <span class="inline-block w-2 h-2 bg-purple-400 rounded-full mr-3 animate-pulse"></span>
          Analyzing Check...
        </h2>
        <p class="text-gray-300">AI is processing your image through multiple detection layers</p>
      </div>

      <!-- Progress Bar -->
      <div class="mb-8">
        <div class="flex justify-between items-center mb-3">
          <span class="text-sm font-semibold text-gray-300">Stage {{ currentStage }} of 8</span>
          <span class="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">{{ progress }}%</span>
        </div>
        <div class="relative w-full h-3 bg-white/10 rounded-full overflow-hidden backdrop-blur-sm">
          <div class="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 animate-pulse"></div>
          <div
            class="relative h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full transition-all duration-700 ease-out shadow-lg shadow-purple-500/50"
            [style.width.%]="progress"
          >
            <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
          </div>
        </div>
      </div>

      <!-- Status Message -->
      <div class="flex items-center space-x-4 p-5 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-purple-500/30 backdrop-blur-sm mb-8">
        <!-- Spinner -->
        <div class="flex-shrink-0">
          <div class="relative w-10 h-10">
            <div class="absolute inset-0 border-4 border-purple-500/30 rounded-full"></div>
            <div class="absolute inset-0 border-4 border-transparent border-t-purple-500 rounded-full animate-spin"></div>
            <div class="absolute inset-2 border-4 border-transparent border-t-blue-400 rounded-full animate-spin-slow"></div>
          </div>
        </div>
        <div class="flex-1">
          <p class="text-base font-medium text-white">{{ statusMessage }}</p>
          <p class="text-xs text-gray-400 mt-1">Processing...</p>
        </div>
      </div>

      <!-- Stage Indicators -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div
          *ngFor="let stage of stages; let i = index"
          [ngClass]="{
            'flex flex-col items-center p-4 rounded-xl transition-all duration-300 border': true,
            'bg-green-500/20 border-green-500/50': i + 1 < currentStage,
            'bg-purple-500/20 border-purple-500/50': i + 1 === currentStage,
            'bg-white/5 border-white/10': i + 1 > currentStage
          }"
        >
          <div
            [ngClass]="{
              'w-12 h-12 rounded-full flex items-center justify-center mb-3 transition-all duration-300 shadow-lg bg-gradient-to-br': true,
              'from-green-500 to-emerald-600': i + 1 < currentStage,
              'from-purple-500 to-pink-600 animate-pulse': i + 1 === currentStage,
              'from-gray-600 to-gray-700': i + 1 > currentStage
            }"
          >
            <svg *ngIf="i + 1 < currentStage" class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path>
            </svg>
            <svg *ngIf="i + 1 === currentStage" class="w-6 h-6 text-white animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span *ngIf="i + 1 > currentStage" class="text-sm font-bold text-gray-400">{{ i + 1 }}</span>
          </div>
          <p
            [ngClass]="{
              'text-xs text-center font-semibold': true,
              'text-green-400': i + 1 < currentStage,
              'text-purple-400': i + 1 === currentStage,
              'text-gray-500': i + 1 > currentStage
            }"
          >
            {{ stage }}
          </p>
        </div>
      </div>
    </div>
  `,
  styles: [
    `
      @keyframes shimmer {
        0% {
          transform: translateX(-100%);
        }
        100% {
          transform: translateX(100%);
        }
      }
      .animate-shimmer {
        animation: shimmer 2s infinite;
      }
      .animate-spin-slow {
        animation: spin 3s linear infinite;
      }
      @keyframes spin {
        to {
          transform: rotate(360deg);
        }
      }
    `,
  ],
})
export class ProgressComponent {
  @Input() progress = 0;
  @Input() currentStage = 0;
  @Input() statusMessage = '';

  stages = ['Validation', 'Preprocessing', 'Metadata', 'OCR', 'Forensics', 'Security', 'Templates', 'ML Analysis'];
}
