# Generated by Django 3.1.5 on 2021-04-29 19:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ProjectConfig',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('timezone', models.CharField(default='Coordinated Universal Time UTC+00', max_length=75)),
                ('date_format', models.CharField(default='dd/mm/yy', max_length=75)),
                ('is_debug', models.BooleanField(default=False, verbose_name='is_debug')),
                ('is_demo_account', models.BooleanField(default=False, verbose_name='is_demo_account')),
                ('is_verified', models.BooleanField(default=False, verbose_name='is_verified')),
                ('primary_username', models.CharField(blank=True, max_length=50, null=True, verbose_name='primary_username')),
                ('name', models.CharField(blank=True, max_length=50, null=True, verbose_name='name')),
                ('phone', models.CharField(blank=True, max_length=15, null=True, verbose_name='phone')),
                ('company_name', models.CharField(blank=True, max_length=50, null=True, verbose_name='company name')),
                ('country', models.CharField(blank=True, max_length=50, null=True, verbose_name='country')),
                ('company_size', models.CharField(blank=True, max_length=30, null=True, verbose_name='company size')),
                ('business_role', models.CharField(blank=True, max_length=50, null=True, verbose_name='business_role')),
            ],
        ),
        migrations.CreateModel(
            name='ProjectUsage',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('primary_username', models.CharField(blank=True, max_length=50, null=True, verbose_name='primary_username')),
                ('n_installer_downloaded', models.IntegerField(default=0, verbose_name='n_installer_downloaded')),
                ('last_installer_download_date', models.DateTimeField(blank=True, null=True)),
            ],
        ),
    ]