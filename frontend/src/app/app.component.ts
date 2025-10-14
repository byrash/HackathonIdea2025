import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UploadComponent } from './components/upload.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, UploadComponent],
  template: `
    <div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <!-- Animated background -->
      <div class="fixed inset-0 overflow-hidden pointer-events-none">
        <div class="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div class="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-indigo-500/10 to-pink-500/10 rounded-full blur-3xl animate-pulse" style="animation-delay: 1s;"></div>
      </div>

      <!-- Header -->
      <header class="relative bg-white/10 backdrop-blur-md border-b border-white/20 shadow-2xl">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div class="flex items-center space-x-4">
            <div class="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                ></path>
              </svg>
            </div>
            <div>
              <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Fraudulent Bank Check Detection</h1>
              <p class="mt-1 text-sm text-gray-300">AI-powered verification using advanced image forensics</p>
            </div>
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <app-upload></app-upload>
      </main>

      <!-- Footer -->
      <footer class="relative mt-16 bg-white/5 backdrop-blur-sm border-t border-white/10">
        <div class="max-w-7xl mx-auto px-4 py-6 text-center">
          <p class="text-sm text-gray-400"><span class="font-semibold text-purple-400">Hackathon Project 2025</span> • For Educational Purposes Only</p>
        </div>
      </footer>
    </div>
  `,
  styles: [],
})
export class AppComponent {
  title = 'Check Fraud Detector';
}
